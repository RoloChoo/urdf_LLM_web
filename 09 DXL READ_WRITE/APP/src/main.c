/************************* (C) COPYRIGHT 2010 ROBOTIS **************************
* File Name          : main.c
* Author             : danceww (modified for bridge)
* Version            : V0.1.0
* Date               : 2026-01-16
* Description        : PC-to-Dynamixel Bridge for CM-530
*******************************************************************************/
#include <stdint.h>
#include "stm32f10x_lib.h"
#include "dynamixel.h"
#include "dxl_hal.h"

/* AX-12A Control Table Addresses (Protocol 1.0) */
#define P_CW_ANGLE_LIMIT_L      6
#define P_CCW_ANGLE_LIMIT_L     8
#define P_TORQUE_ENABLE         24
#define P_GOAL_POSITION_L       30
#define P_GOAL_POSITION_H       31
#define P_MOVING_SPEED_L        32
#define P_TORQUE_LIMIT_L        34
#define P_PRESENT_POSITION_L    36
#define P_PRESENT_POSITION_H    37
#define P_PRESENT_VOLTAGE       42
#define P_PRESENT_TEMPERATURE   43
#define P_MOVING                46

/* Hardware Defines */
#define PORT_ENABLE_TXD         GPIOB
#define PORT_ENABLE_RXD         GPIOB
#define PIN_ENABLE_TXD          GPIO_Pin_4
#define PIN_ENABLE_RXD          GPIO_Pin_5
#define PIN_DXL_TXD             GPIO_Pin_6
#define PIN_DXL_RXD             GPIO_Pin_7
#define PIN_PC_TXD              GPIO_Pin_10
#define PIN_PC_RXD              GPIO_Pin_11

#define USART_DXL               0
#define USART_PC                2

#define word                    u16
#define byte                    u8

/* Joint Configuration */
typedef struct {
    byte    id;
    int16_t zero_tick;      // home position in ticks
    int8_t  direction;      // +1 or -1
    word    min_tick;       // safety clamp min
    word    max_tick;       // safety clamp max
    word    speed_limit;    // 0~1023
    word    torque_limit;   // 0~1023
} JointCfg;

/* TODO: 실제 로봇에 맞게 수정 */
static const JointCfg g_joints[] = {
    { .id=1, .zero_tick=512, .direction=+1, .min_tick=100, .max_tick=924, .speed_limit=200, .torque_limit=600 },
    { .id=2, .zero_tick=512, .direction=-1, .min_tick=100, .max_tick=924, .speed_limit=200, .torque_limit=600 },
    { .id=3, .zero_tick=512, .direction=+1, .min_tick=100, .max_tick=924, .speed_limit=200, .torque_limit=600 },
};
#define NUM_JOINTS (sizeof(g_joints)/sizeof(g_joints[0]))

/* Global Variables */
volatile byte   gbpRxInterruptBuffer[256];
volatile byte   gbRxBufferWritePointer, gbRxBufferReadPointer;
volatile vu32   gwTimingDelay, gw1msCounter;
u32             Baudrate_DXL = 1000000;
u32             Baudrate_PC  = 57600;
vu16            CCR1_Val = 100;
vu32            capture = 0;

/* Function Prototypes */
void RCC_Configuration(void);
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void SysTick_Configuration(void);
void Timer_Configuration(void);
void USART_Configuration(u8, u32);
void mDelay(u32);
void TxDString(byte*);
void TxDWord16(word);
void TxDByte16(byte);
void TxDByte_PC(byte);
void TxDDecimal(word);

/* ========== Utility Functions ========== */

static const JointCfg* find_joint(byte id)
{
    int i;
    for(i = 0; i < (int)NUM_JOINTS; i++) {
        if(g_joints[i].id == id) return &g_joints[i];
    }
    return 0;
}

static int deg10_to_tick(const JointCfg* j, int deg10)
{
    // 1 tick ≈ 0.29°, so tick_offset = deg10 * 10 / 29
    int32_t off = (int32_t)deg10 * 10 / 29;
    int32_t tick = (int32_t)j->zero_tick + (int32_t)j->direction * off;
    if(tick < (int32_t)j->min_tick) tick = j->min_tick;
    if(tick > (int32_t)j->max_tick) tick = j->max_tick;
    return (int)tick;
}

static void set_goal_tick(byte id, word tick)
{
    if(tick > 1023) tick = 1023;
    dxl_write_word(id, P_GOAL_POSITION_L, tick);
}

/* ========== Joint Initialization ========== */

static void init_joint(const JointCfg* j)
{
    word cw, ccw;
    
    // Check if in wheel mode, restore to joint mode
    cw  = dxl_read_word(j->id, P_CW_ANGLE_LIMIT_L);
    ccw = dxl_read_word(j->id, P_CCW_ANGLE_LIMIT_L);
    
    if(cw == 0 && ccw == 0) {
        dxl_write_word(j->id, P_CW_ANGLE_LIMIT_L, 0);
        dxl_write_word(j->id, P_CCW_ANGLE_LIMIT_L, 1023);
        mDelay(10);
    }
    
    // Set speed and torque limits
    dxl_write_word(j->id, P_MOVING_SPEED_L, j->speed_limit);
    dxl_write_word(j->id, P_TORQUE_LIMIT_L, j->torque_limit);
    
    // Enable torque
    dxl_write_byte(j->id, P_TORQUE_ENABLE, 1);
}

static void init_all_joints(void)
{
    int i;
    for(i = 0; i < (int)NUM_JOINTS; i++) {
        init_joint(&g_joints[i]);
        mDelay(10);
    }
}

/* ========== PC Communication ========== */

static int PC_ReadByteNB(byte* out)
{
    if(USART_GetFlagStatus(USART3, USART_FLAG_RXNE) == RESET)
        return 0;
    *out = (byte)USART_ReceiveData(USART3);
    return 1;
}

static int PC_ReadLine(char* line, int maxlen)
{
    static char buf[64];
    static int  len = 0;
    byte c;
    
    while(PC_ReadByteNB(&c)) {
        if(c == '\r' || c == '\n') {
            if(len == 0) continue;
            buf[len] = 0;
            // copy out
            int i;
            for(i = 0; i < maxlen - 1 && buf[i]; i++) 
                line[i] = buf[i];
            line[i] = 0;
            len = 0;
            return 1;
        } else {
            if(len < (int)sizeof(buf) - 1) 
                buf[len++] = (char)c;
        }
    }
    return 0;
}

/* ========== Simple Parser (no sscanf) ========== */

static int parse_int(const char** pp)
{
    const char* p = *pp;
    int sign = 1;
    int val = 0;
    
    // skip spaces
    while(*p == ' ') p++;
    
    // sign
    if(*p == '-') { sign = -1; p++; }
    else if(*p == '+') { p++; }
    
    // digits
    while(*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    
    *pp = p;
    return sign * val;
}

/* ========== Status Output ========== */

void TxDDecimal(word val)
{
    char buf[6];
    int i = 5;
    buf[i--] = 0;
    
    if(val == 0) {
        TxDByte_PC('0');
        return;
    }
    
    while(val > 0 && i >= 0) {
        buf[i--] = '0' + (val % 10);
        val /= 10;
    }
    TxDString((byte*)&buf[i+1]);
}

static void print_status(byte id, word goal_tick)
{
    word pos;
    byte v, t;
    
    pos = dxl_read_word(id, P_PRESENT_POSITION_L);
    v   = dxl_read_byte(id, P_PRESENT_VOLTAGE);
    t   = dxl_read_byte(id, P_PRESENT_TEMPERATURE);
    
    // Format: STAT,<id>,<goal>,<pos>,<volt>,<temp>
    TxDString((byte*)"STAT,");
    TxDDecimal(id);
    TxDByte_PC(',');
    TxDDecimal(goal_tick);
    TxDByte_PC(',');
    TxDDecimal(pos);
    TxDByte_PC(',');
    TxDDecimal(v);
    TxDByte_PC(',');
    TxDDecimal(t);
    TxDString((byte*)"\r\n");
}

static void print_error_if_any(void)
{
    if(dxl_get_rxpacket_error(ERRBIT_VOLTAGE) == 1)
        TxDString((byte*)"ERR,VOLTAGE\r\n");
    if(dxl_get_rxpacket_error(ERRBIT_ANGLE) == 1)
        TxDString((byte*)"ERR,ANGLE_LIMIT\r\n");
    if(dxl_get_rxpacket_error(ERRBIT_OVERHEAT) == 1)
        TxDString((byte*)"ERR,OVERHEAT\r\n");
    if(dxl_get_rxpacket_error(ERRBIT_OVERLOAD) == 1)
        TxDString((byte*)"ERR,OVERLOAD\r\n");
}

/* ========== Command Handler ========== */

static void handle_line(const char* s)
{
    const char* p = s;
    char cmd;
    int id_i, val;
    byte id;
    const JointCfg* j;
    word tick;
    
    // skip leading spaces
    while(*p == ' ') p++;
    if(*p == 0) return;
    
    cmd = *p++;
    
    switch(cmd) {
    
    case 'Q':  // Q <id> - Query status
    case 'q':
        id_i = parse_int(&p);
        id = (byte)id_i;
        print_status(id, 0);
        print_error_if_any();
        break;
        
    case 'T':  // T <id> <tick> - Set goal by tick
    case 't':
        id_i = parse_int(&p);
        val  = parse_int(&p);
        id = (byte)id_i;
        tick = (word)val;
        
        j = find_joint(id);
        if(j) {
            if(tick < j->min_tick) tick = j->min_tick;
            if(tick > j->max_tick) tick = j->max_tick;
        }
        
        set_goal_tick(id, tick);
        TxDString((byte*)"OK,T,");
        TxDDecimal(id);
        TxDByte_PC(',');
        TxDDecimal(tick);
        TxDString((byte*)"\r\n");
        break;
        
    case 'D':  // D <id> <deg10> - Set goal by 0.1 degree
    case 'd':
        id_i = parse_int(&p);
        val  = parse_int(&p);
        id = (byte)id_i;
        
        j = find_joint(id);
        if(!j) {
            TxDString((byte*)"ERR,UNKNOWN_ID\r\n");
            break;
        }
        
        tick = (word)deg10_to_tick(j, val);
        set_goal_tick(id, tick);
        TxDString((byte*)"OK,D,");
        TxDDecimal(id);
        TxDByte_PC(',');
        TxDDecimal(tick);
        TxDString((byte*)"\r\n");
        break;
        
    case 'H':  // H - Home all joints (go to zero_tick)
    case 'h':
        {
            int i;
            for(i = 0; i < (int)NUM_JOINTS; i++) {
                set_goal_tick(g_joints[i].id, g_joints[i].zero_tick);
            }
            TxDString((byte*)"OK,HOME\r\n");
        }
        break;
        
    case 'E':  // E <id> <0|1> - Torque enable/disable
    case 'e':
        id_i = parse_int(&p);
        val  = parse_int(&p);
        id = (byte)id_i;
        dxl_write_byte(id, P_TORQUE_ENABLE, (val ? 1 : 0));
        TxDString((byte*)"OK,E,");
        TxDDecimal(id);
        TxDByte_PC(',');
        TxDDecimal(val ? 1 : 0);
        TxDString((byte*)"\r\n");
        break;
        
    case 'S':  // S <id> <speed> <torque> - Set limits
    case 's':
        id_i = parse_int(&p);
        {
            int spd = parse_int(&p);
            int trq = parse_int(&p);
            id = (byte)id_i;
            if(spd > 1023) spd = 1023;
            if(trq > 1023) trq = 1023;
            dxl_write_word(id, P_MOVING_SPEED_L, (word)spd);
            dxl_write_word(id, P_TORQUE_LIMIT_L, (word)trq);
            TxDString((byte*)"OK,S\r\n");
        }
        break;
        
    case 'P':  // P - Ping all and report
    case 'p':
        {
            int i;
            TxDString((byte*)"PING");
            for(i = 0; i < (int)NUM_JOINTS; i++) {
                dxl_ping(g_joints[i].id);
                if(dxl_get_result() == COMM_RXSUCCESS) {
                    TxDByte_PC(',');
                    TxDDecimal(g_joints[i].id);
                }
            }
            TxDString((byte*)"\r\n");
        }
        break;
        
    case '?':  // Help
        TxDString((byte*)"Commands:\r\n");
        TxDString((byte*)"  Q <id>        - Query status\r\n");
        TxDString((byte*)"  T <id> <tick> - Set goal tick\r\n");
        TxDString((byte*)"  D <id> <deg10>- Set goal 0.1deg\r\n");
        TxDString((byte*)"  H             - Home all\r\n");
        TxDString((byte*)"  E <id> <0|1>  - Torque on/off\r\n");
        TxDString((byte*)"  S <id> <spd> <trq> - Set limits\r\n");
        TxDString((byte*)"  P             - Ping all\r\n");
        break;
        
    default:
        TxDString((byte*)"ERR,UNKNOWN_CMD\r\n");
        break;
    }
}

/* ========== Main ========== */

int main(void)
{
    char line[64];
    
    RCC_Configuration();
    NVIC_Configuration();
    GPIO_Configuration();
    SysTick_Configuration();
    Timer_Configuration();
    
    dxl_initialize(0, 1);
    USART_Configuration(USART_PC, Baudrate_PC);
    
    mDelay(500);  // wait for servos to power up
    
    TxDString((byte*)"\r\n================================\r\n");
    TxDString((byte*)"CM-530 DXL Bridge v0.1\r\n");
    TxDString((byte*)"Type ? for help\r\n");
    TxDString((byte*)"================================\r\n");
    
    init_all_joints();
    TxDString((byte*)"Joints initialized: ");
    TxDDecimal(NUM_JOINTS);
    TxDString((byte*)"\r\nREADY\r\n");
    
    while(1)
    {
        if(PC_ReadLine(line, sizeof(line))) {
            handle_line(line);
        }
    }
    
    return 0;
}

/* ========== Hardware Configuration ========== */

void RCC_Configuration(void)
{
    ErrorStatus HSEStartUpStatus;
    RCC_DeInit();
    RCC_HSEConfig(RCC_HSE_ON);
    HSEStartUpStatus = RCC_WaitForHSEStartUp();

    if(HSEStartUpStatus == SUCCESS)
    {
        FLASH_PrefetchBufferCmd(FLASH_PrefetchBuffer_Enable);
        FLASH_SetLatency(FLASH_Latency_2);
        RCC_HCLKConfig(RCC_SYSCLK_Div1);
        RCC_PCLK2Config(RCC_HCLK_Div1);
        RCC_PCLK1Config(RCC_HCLK_Div2);
        RCC_PLLConfig(RCC_PLLSource_HSE_Div1, RCC_PLLMul_9);
        RCC_PLLCmd(ENABLE);
        while(RCC_GetFlagStatus(RCC_FLAG_PLLRDY) == RESET) {}
        RCC_SYSCLKConfig(RCC_SYSCLKSource_PLLCLK);
        while(RCC_GetSYSCLKSource() != 0x08) {}
    }

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1 | RCC_APB2Periph_GPIOB, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3 | RCC_APB1Periph_TIM2, ENABLE);
    PWR_BackupAccessCmd(ENABLE);
}

void NVIC_Configuration(void)
{
    NVIC_InitTypeDef NVIC_InitStructure;

    #ifdef VECT_TAB_RAM
        NVIC_SetVectorTable(NVIC_VectTab_RAM, 0x0);
    #else
        NVIC_SetVectorTable(NVIC_VectTab_FLASH, 0x3000);
    #endif

    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);

    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQChannel;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);

    NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQChannel;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
}

void GPIO_Configuration(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_StructInit(&GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = PIN_ENABLE_TXD | PIN_ENABLE_RXD;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = PIN_DXL_RXD | PIN_PC_RXD;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = PIN_DXL_TXD | PIN_PC_TXD;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

    GPIO_PinRemapConfig(GPIO_Remap_USART1, ENABLE);
    GPIO_PinRemapConfig(GPIO_Remap_SWJ_Disable, ENABLE);

    GPIO_ResetBits(PORT_ENABLE_TXD, PIN_ENABLE_TXD);
    GPIO_SetBits(PORT_ENABLE_RXD, PIN_ENABLE_RXD);
}

void USART_Configuration(u8 PORT, u32 baudrate)
{
    USART_InitTypeDef USART_InitStructure;
    USART_StructInit(&USART_InitStructure);

    USART_InitStructure.USART_BaudRate = baudrate;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;

    if(PORT == USART_DXL)
    {
        USART_DeInit(USART1);
        mDelay(10);
        USART_Init(USART1, &USART_InitStructure);
        USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
        USART_Cmd(USART1, ENABLE);
    }
    else if(PORT == USART_PC)
    {
        USART_DeInit(USART3);
        mDelay(10);
        USART_Init(USART3, &USART_InitStructure);
        USART_Cmd(USART3, ENABLE);
    }
}

void SysTick_Configuration(void)
{
    SysTick_SetReload(9000);
    SysTick_ITConfig(ENABLE);
}

void Timer_Configuration(void)
{
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_OCInitTypeDef TIM_OCInitStructure;

    TIM_TimeBaseStructInit(&TIM_TimeBaseStructure);
    TIM_OCStructInit(&TIM_OCInitStructure);
    TIM_DeInit(TIM2);

    TIM_TimeBaseStructure.TIM_Period = 65535;
    TIM_TimeBaseStructure.TIM_Prescaler = 0;
    TIM_TimeBaseStructure.TIM_ClockDivision = 0;
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);

    TIM_PrescalerConfig(TIM2, 722, TIM_PSCReloadMode_Immediate);

    TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Timing;
    TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Disable;
    TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
    TIM_OCInitStructure.TIM_Pulse = CCR1_Val;
    TIM_OC1Init(TIM2, &TIM_OCInitStructure);
    TIM_OC1PreloadConfig(TIM2, TIM_OCPreload_Disable);

    TIM_ITConfig(TIM2, TIM_IT_CC1, ENABLE);
    TIM_Cmd(TIM2, ENABLE);
}

void mDelay(u32 nTime)
{
    SysTick_CounterCmd(SysTick_Counter_Enable);
    gwTimingDelay = nTime;
    while(gwTimingDelay != 0);
    SysTick_CounterCmd(SysTick_Counter_Disable);
    SysTick_CounterCmd(SysTick_Counter_Clear);
}

void __ISR_DELAY(void)
{
    if(gwTimingDelay != 0x00)
        gwTimingDelay--;
}

void TimerInterrupt_1ms(void)
{
    if(TIM_GetITStatus(TIM2, TIM_IT_CC1) != RESET)
    {
        TIM_ClearITPendingBit(TIM2, TIM_IT_CC1);
        capture = TIM_GetCapture1(TIM2);
        TIM_SetCompare1(TIM2, capture + CCR1_Val);
        if(gw1msCounter > 0)
            gw1msCounter--;
    }
}

void RxD0Interrupt(void)
{
    if(USART_GetITStatus(USART1, USART_IT_RXNE) != RESET)
        gbpRxInterruptBuffer[gbRxBufferWritePointer++] = USART_ReceiveData(USART1);
}

void TxDByte_PC(byte bTxdData)
{
    USART_SendData(USART3, bTxdData);
    while(USART_GetFlagStatus(USART3, USART_FLAG_TC) == RESET);
}

void TxDString(byte *bData)
{
    while(*bData)
        TxDByte_PC(*bData++);
}

void TxDByte16(byte bSentData)
{
    byte bTmp;
    bTmp = ((byte)(bSentData >> 4) & 0x0f) + (byte)'0';
    if(bTmp > '9') bTmp += 7;
    TxDByte_PC(bTmp);
    bTmp = (byte)(bSentData & 0x0f) + (byte)'0';
    if(bTmp > '9') bTmp += 7;
    TxDByte_PC(bTmp);
}

void TxDWord16(word wSentData)
{
    TxDByte16((wSentData >> 8) & 0xff);
    TxDByte16(wSentData & 0xff);
}

/* ========== DXL HAL Functions (Required by dynamixel.c) ========== */

void USART1_Configuration(u32 baudrate)
{
    USART_Configuration(USART_DXL, baudrate);
}

void DisableUSART1(void)
{
    USART_Cmd(USART1, DISABLE);
}

void ClearBuffer256(void)
{
    gbRxBufferReadPointer = 0;
    gbRxBufferWritePointer = 0;
}

byte CheckNewArrive(void)
{
    return (gbRxBufferReadPointer != gbRxBufferWritePointer) ? 1 : 0;
}

void TxDByte_DXL(byte bTxdData)
{
    GPIO_ResetBits(PORT_ENABLE_RXD, PIN_ENABLE_RXD);
    GPIO_SetBits(PORT_ENABLE_TXD, PIN_ENABLE_TXD);

    USART_SendData(USART1, bTxdData);
    while(USART_GetFlagStatus(USART1, USART_FLAG_TC) == RESET);

    GPIO_ResetBits(PORT_ENABLE_TXD, PIN_ENABLE_TXD);
    GPIO_SetBits(PORT_ENABLE_RXD, PIN_ENABLE_RXD);
}

byte RxDByte_DXL(void)
{
    while(gbRxBufferReadPointer == gbRxBufferWritePointer);
    return gbpRxInterruptBuffer[gbRxBufferReadPointer++];
}

void StartDiscount(s32 StartTime)
{
    gw1msCounter = StartTime;
}

byte CheckTimeOut(void)
{
    return (gw1msCounter == 0) ? 1 : 0;
}