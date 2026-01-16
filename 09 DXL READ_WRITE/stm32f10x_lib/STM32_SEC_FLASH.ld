/*
  Fixed Linker Script for STM32F10x devices in FLASH mode
  Prevents .init overlaps .data error
  Based on RAISONANCE 2005 template
*/

/* Sections Definitions */

SECTIONS
{
    /* 벡터 테이블을 FLASH의 절대 시작 주소에 고정 */
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } >FLASH
    
    /* Program code section - goes to FLASH */
    .text :
    {
        . = ALIGN(4);
        
        *(.text)                   /* remaining code */
        *(.text.*)                 /* remaining code */
        *(.rodata)                 /* read-only data (constants) */
        *(.rodata*)
        *(.glue_7)
        *(.glue_7t)

        /* Constructor/Destructor sections - MOVED INSIDE .text to prevent overlap */
        . = ALIGN(4);
        KEEP(*(.init))
        KEEP(*(.fini))
        KEEP(*(.preinit_array))
        KEEP(*(.init_array))
        KEEP(*(.fini_array))
        
        /* ARM unwind tables - 심볼을 섹션 안에서 감싸야 정확함 */
        *(.ARM.extab*)
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;

        . = ALIGN(4);
        _etext = .;        /* 코드 끝 */
        _sidata = _etext;  /* startup이 .data 초기화할 때 사용 */
    } >FLASH

    /* Initialized data section - executes from RAM, stored in FLASH */
    .data : AT ( _sidata )
    {
        . = ALIGN(4);
        _sdata = .;
        
        *(.data)
        *(.data.*)

        . = ALIGN(4);
        _edata = .;
    } >RAM

    /* Uninitialized data section */
    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        
        *(.bss)
        *(.bss.*)
        *(COMMON)
        
        . = ALIGN(4);
        _ebss = .;
    } >RAM
    
    PROVIDE ( end = _ebss );
    PROVIDE ( _end = _ebss );
    
    /* User stack section */
    ._usrstack :
    {
        . = ALIGN(4);
        _susrstack = .;
        
        . = . + _Minimum_Stack_Size ;
        
        . = ALIGN(4);
        _eusrstack = .;
    } >RAM

    /* FLASH Bank1 */
    .b1text :
    {
        *(.b1text)
        *(.b1rodata)
        *(.b1rodata*)
    } >FLASHB1
    
    /* EXTMEM Banks */
    .eb0text :
    {
        *(.eb0text)
        *(.eb0rodata)
        *(.eb0rodata*)
    } >EXTMEMB0
    
    .eb1text :
    {
        *(.eb1text)
        *(.eb1rodata)
        *(.eb1rodata*)
    } >EXTMEMB1
    
    .eb2text :
    {
        *(.eb2text)
        *(.eb2rodata)
        *(.eb2rodata*)
    } >EXTMEMB2
    
    .eb3text :
    {
        *(.eb3text)
        *(.eb3rodata)
        *(.eb3rodata*)
    } >EXTMEMB3

    /* Debugging sections */
    .stab          0 : { *(.stab) }
    .stabstr       0 : { *(.stabstr) }
    .stab.excl     0 : { *(.stab.excl) }
    .stab.exclstr  0 : { *(.stab.exclstr) }
    .stab.index    0 : { *(.stab.index) }
    .stab.indexstr 0 : { *(.stab.indexstr) }
    .comment       0 : { *(.comment) }
    
    /* DWARF debug sections */
    .debug          0 : { *(.debug) }
    .line           0 : { *(.line) }
    .debug_srcinfo  0 : { *(.debug_srcinfo) }
    .debug_sfnames  0 : { *(.debug_sfnames) }
    .debug_aranges  0 : { *(.debug_aranges) }
    .debug_pubnames 0 : { *(.debug_pubnames) }
    .debug_info     0 : { *(.debug_info .gnu.linkonce.wi.*) }
    .debug_abbrev   0 : { *(.debug_abbrev) }
    .debug_line     0 : { *(.debug_line) }
    .debug_frame    0 : { *(.debug_frame) }
    .debug_str      0 : { *(.debug_str) }
    .debug_loc      0 : { *(.debug_loc) }
    .debug_macinfo  0 : { *(.debug_macinfo) }
    .debug_weaknames 0 : { *(.debug_weaknames) }
    .debug_funcnames 0 : { *(.debug_funcnames) }
    .debug_typenames 0 : { *(.debug_typenames) }
    .debug_varnames  0 : { *(.debug_varnames) }
}