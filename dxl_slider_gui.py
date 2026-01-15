import tkinter as tk
from tkinter import ttk, messagebox

from dynamixel_sdk import PortHandler, PacketHandler

# (선택) 포트 자동 나열용. 없으면 수동 입력만 가능.
try:
    from serial.tools import list_ports
except Exception:
    list_ports = None

# =====================
# DYNAMIXEL (AX-12A / Protocol 1.0) 기본 주소
# =====================
PROTOCOL_VERSION = 1.0

ADDR_TORQUE_ENABLE     = 24
ADDR_GOAL_POSITION     = 30
ADDR_MOVING_SPEED      = 32
ADDR_TORQUE_LIMIT      = 34
ADDR_PRESENT_POSITION  = 36

POS_MIN = 0
POS_MAX = 1023


# =====================
# DXL 세션 (연결/해제 가능하게)
# =====================
class DxlSession:
    def __init__(self, protocol_version: float = 1.0):
        self.packet = PacketHandler(protocol_version)
        self.port = None
        self.connected = False
        self.device = ""
        self.baud = 0

    def connect(self, device: str, baud: int) -> tuple[bool, str]:
        """openPort()/setBaudRate()는 성공/실패를 bool로 준다. :contentReference[oaicite:4]{index=4}"""
        self.disconnect()

        self.device = device
        self.baud = baud
        self.port = PortHandler(device)

        if not self.port.openPort():
            self.connected = False
            return False, f"openPort() 실패: {device}"

        # openPort() 후 필요한 baud로 setBaudRate() 호출하는 패턴이 공식 예제에 있다. :contentReference[oaicite:5]{index=5}
        if not self.port.setBaudRate(baud):
            try:
                self.port.closePort()
            except Exception:
                pass
            self.connected = False
            return False, f"setBaudRate() 실패: {baud}"

        self.connected = True
        return True, f"Connected ({device}, {baud})"

    def disconnect(self):
        if self.port is not None:
            try:
                self.port.closePort()
            except Exception:
                pass
        self.port = None
        self.connected = False

    def explain_error(self, comm_result, dxl_error) -> str:
        msgs = []
        if comm_result is not None and comm_result != 0:
            msgs.append(self.packet.getTxRxResult(comm_result))
        if dxl_error is not None and dxl_error != 0:
            msgs.append(self.packet.getRxPacketError(dxl_error))
        return " / ".join(msgs)

    def write1(self, dxl_id: int, addr: int, val: int) -> str:
        if not self.connected or self.port is None:
            return "Not connected"
        comm_result, dxl_error = self.packet.write1ByteTxRx(self.port, dxl_id, addr, val)
        return self.explain_error(comm_result, dxl_error)

    def write2(self, dxl_id: int, addr: int, val: int) -> str:
        if not self.connected or self.port is None:
            return "Not connected"
        comm_result, dxl_error = self.packet.write2ByteTxRx(self.port, dxl_id, addr, val)
        return self.explain_error(comm_result, dxl_error)

    def read2(self, dxl_id: int, addr: int) -> tuple[int | None, str]:
        if not self.connected or self.port is None:
            return None, "Not connected"
        val, comm_result, dxl_error = self.packet.read2ByteTxRx(self.port, dxl_id, addr)
        return val, self.explain_error(comm_result, dxl_error)


# =====================
# GUI
# =====================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DYNAMIXEL Slider Controller (Connect/Disconnect UI)")
        self.geometry("820x560")

        self.dxl = DxlSession(PROTOCOL_VERSION)

        # UI 상태
        self.status_text = tk.StringVar(value="Disconnected")
        self.error_text = tk.StringVar(value="")
        self.connected_var = tk.BooleanVar(value=False)

        # 연결 설정
        self.port_var = tk.StringVar(value="")
        self.baud_var = tk.StringVar(value="1000000")
        self.ids_var = tk.StringVar(value="1")  # "1,2,3" 형태

        # 제어 옵션
        self.torque_on = tk.BooleanVar(value=False)
        self.send_on_drag = tk.BooleanVar(value=False)
        self.speed_var = tk.IntVar(value=200)
        self.torque_limit_var = tk.IntVar(value=512)

        # 모터 UI
        self.rows = {}         # id -> widgets
        self.poll_job = None

        self._build_ui()
        self.refresh_ports()
        self._set_connected_ui(False)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI 구성 ----------
    def _build_ui(self):
        # 상단: 연결 바
        conn = ttk.LabelFrame(self, text="Connection", padding=10)
        conn.pack(fill="x", padx=10, pady=10)

        # 상태 표시 원(회색/초록): Canvas로 원 그리는 게 제일 쉬움 :contentReference[oaicite:6]{index=6}
        self.indicator = tk.Canvas(conn, width=18, height=18, highlightthickness=0)
        self.indicator.grid(row=0, column=0, padx=(0, 8))
        self.indicator_oval = self.indicator.create_oval(2, 2, 16, 16, fill="gray", outline="")

        ttk.Label(conn, textvariable=self.status_text, width=18).grid(row=0, column=1, sticky="w")

        ttk.Label(conn, text="Port").grid(row=0, column=2, padx=(16, 6), sticky="e")
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, width=28)
        self.port_combo.grid(row=0, column=3, sticky="w")

        ttk.Button(conn, text="Refresh", command=self.refresh_ports).grid(row=0, column=4, padx=6)

        ttk.Label(conn, text="Baud").grid(row=0, column=5, padx=(16, 6), sticky="e")
        self.baud_entry = ttk.Entry(conn, textvariable=self.baud_var, width=10)
        self.baud_entry.grid(row=0, column=6, sticky="w")

        ttk.Label(conn, text="IDs").grid(row=0, column=7, padx=(16, 6), sticky="e")
        self.ids_entry = ttk.Entry(conn, textvariable=self.ids_var, width=16)
        self.ids_entry.grid(row=0, column=8, sticky="w")

        self.connect_btn = ttk.Button(conn, text="Connect", command=self.toggle_connect)
        self.connect_btn.grid(row=0, column=9, padx=(16, 0))

        # 에러/안내 텍스트
        err = ttk.Label(conn, textvariable=self.error_text, foreground="red")
        err.grid(row=1, column=0, columnspan=10, sticky="w", pady=(6, 0))

        # 공통 설정
        common = ttk.LabelFrame(self, text="Common Settings", padding=10)
        common.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Checkbutton(common, text="Torque ON", variable=self.torque_on, command=self.apply_torque_all).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(common, text="드래그 중에도 계속 보내기", variable=self.send_on_drag).grid(row=0, column=1, sticky="w", padx=12)
        ttk.Button(common, text="EMERGENCY STOP (Torque OFF)", command=self.emergency_stop).grid(row=0, column=2, sticky="e")

        ttk.Label(common, text="Moving Speed (0=MAX, 1~1023)").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.speed_scale = ttk.Scale(common, from_=0, to=1023, orient="horizontal",
                                     command=lambda v: self.speed_var.set(int(float(v))))
        self.speed_scale.set(self.speed_var.get())
        self.speed_scale.grid(row=1, column=1, sticky="ew", padx=10, pady=(10, 0))
        ttk.Label(common, textvariable=self.speed_var, width=6).grid(row=1, column=2, sticky="w", pady=(10, 0))
        self.speed_apply_btn = ttk.Button(common, text="Apply", command=self.apply_speed_all)
        self.speed_apply_btn.grid(row=1, column=3, padx=(10, 0), pady=(10, 0))

        ttk.Label(common, text="Torque Limit (0~1023)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.tl_scale = ttk.Scale(common, from_=0, to=1023, orient="horizontal",
                                  command=lambda v: self.torque_limit_var.set(int(float(v))))
        self.tl_scale.set(self.torque_limit_var.get())
        self.tl_scale.grid(row=2, column=1, sticky="ew", padx=10, pady=(10, 0))
        ttk.Label(common, textvariable=self.torque_limit_var, width=6).grid(row=2, column=2, sticky="w", pady=(10, 0))
        self.tl_apply_btn = ttk.Button(common, text="Apply", command=self.apply_torque_limit_all)
        self.tl_apply_btn.grid(row=2, column=3, padx=(10, 0), pady=(10, 0))

        common.columnconfigure(1, weight=1)

        # 모터 패널
        self.motors_frame = ttk.LabelFrame(self, text="Motors", padding=10)
        self.motors_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._build_motors([])

    # ---------- 포트 탐색 ----------
    def refresh_ports(self):
        # pyserial이 있으면 comports로 나열 가능. :contentReference[oaicite:7]{index=7}
        if list_ports is None:
            self.error_text.set("pyserial 없음: 포트 자동목록 불가 (pip install pyserial). 포트를 직접 입력하세요.")
            return

        ports = list(list_ports.comports())
        values = [p.device for p in ports]  # Windows: COM3, Linux: /dev/ttyUSB0 등
        self.port_combo["values"] = values

        # 아무것도 안 잡혀있고 포트가 1개면 자동 선택
        if (not self.port_var.get()) and len(values) == 1:
            self.port_var.set(values[0])

        if len(values) == 0:
            self.error_text.set("포트가 안 잡힘. U2D2/USB2Serial 연결 확인 or 권한(리눅스) 확인.")
        else:
            # 에러 문구는 강제로 지우지 말고, 연결 성공 시에만 지우는 편이 디버깅에 좋음
            pass

    # ---------- 연결 토글 ----------
    def toggle_connect(self):
        if self.dxl.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        port = self.port_var.get().strip()
        if not port:
            self.error_text.set("Port가 비어있음. 예: COM3 또는 /dev/ttyUSB0")
            return

        try:
            baud = int(self.baud_var.get().strip())
        except ValueError:
            self.error_text.set("Baud가 숫자가 아님. 예: 1000000")
            return

        ids = self.parse_ids(self.ids_var.get())
        if not ids:
            self.error_text.set("IDs가 비어있거나 파싱 실패. 예: 1 또는 1,2,3")
            return

        ok, msg = self.dxl.connect(port, baud)
        if not ok:
            # 연결 실패해도 UI는 유지(요청사항)
            self._set_indicator(False)  # 회색 유지
            self.status_text.set("Disconnected")
            self.error_text.set(msg)
            return

        # 연결 성공
        self._set_connected_ui(True)
        self.error_text.set("")  # 성공이면 에러 클리어
        self.status_text.set("Connected")
        self._set_indicator(True)

        # 모터 UI 생성/갱신
        self._build_motors(ids)

        # 공통 설정 적용 (원치 않으면 주석 처리해도 됨)
        self.apply_speed_all()
        self.apply_torque_limit_all()
        self.apply_torque_all()

        self.start_polling()

    def disconnect(self):
        self.stop_polling()

        # 연결 끊기 전에 토크 OFF로 안전하게
        try:
            self.torque_on.set(False)
            self.apply_torque_all()
        except Exception:
            pass

        self.dxl.disconnect()
        self._set_connected_ui(False)
        self.status_text.set("Disconnected")
        self._set_indicator(False)

    def _set_connected_ui(self, connected: bool):
        # 위젯 상태 토글
        if connected:
            self.connect_btn.configure(text="Disconnect")
            self.port_combo.configure(state="disabled")
            self.baud_entry.configure(state="disabled")
            self.ids_entry.configure(state="disabled")
            self.speed_apply_btn.configure(state="normal")
            self.tl_apply_btn.configure(state="normal")
            self.speed_scale.state(["!disabled"])
            self.tl_scale.state(["!disabled"])
        else:
            self.connect_btn.configure(text="Connect")
            self.port_combo.configure(state="normal")
            self.baud_entry.configure(state="normal")
            self.ids_entry.configure(state="normal")
            self.speed_apply_btn.configure(state="disabled")
            self.tl_apply_btn.configure(state="disabled")
            self.speed_scale.state(["disabled"])
            self.tl_scale.state(["disabled"])

        # 모터 슬라이더도 on/off
        for _id, w in self.rows.items():
            if connected:
                w["scale"].state(["!disabled"])
            else:
                w["scale"].state(["disabled"])
                w["present"].set("present: -")

    def _set_indicator(self, connected: bool):
        # 요청: 초록/회색
        self.indicator.itemconfig(self.indicator_oval, fill=("green" if connected else "gray"))

    # ---------- IDs 파싱 ----------
    @staticmethod
    def parse_ids(text: str) -> list[int]:
        parts = [p.strip() for p in text.replace(" ", "").split(",") if p.strip() != ""]
        ids = []
        for p in parts:
            try:
                ids.append(int(p))
            except ValueError:
                return []
        # 중복 제거 + 정렬
        ids = sorted(list(set(ids)))
        return ids

    # ---------- 모터 UI 생성 ----------
    def _build_motors(self, ids: list[int]):
        # 기존 위젯 삭제
        for child in self.motors_frame.winfo_children():
            child.destroy()
        self.rows.clear()

        if not ids:
            ttk.Label(self.motors_frame, text="Disconnected. Connect 후 IDs에 맞게 슬라이더가 생성됩니다.").pack(anchor="w")
            return

        for dxl_id in ids:
            row = ttk.Frame(self.motors_frame)
            row.pack(fill="x", pady=6)

            ttk.Label(row, text=f"ID {dxl_id}", width=8).pack(side="left")

            goal_var = tk.IntVar(value=512)
            present_var = tk.StringVar(value="present: -")

            scale = ttk.Scale(row, from_=POS_MIN, to=POS_MAX, orient="horizontal")
            scale.set(goal_var.get())
            scale.pack(side="left", fill="x", expand=True, padx=10)

            ttk.Label(row, textvariable=goal_var, width=6).pack(side="left")
            ttk.Label(row, textvariable=present_var, width=16).pack(side="left", padx=(10, 0))

            def on_move(v, _id=dxl_id, _var=goal_var):
                _var.set(int(float(v)))
                if self.dxl.connected and self.send_on_drag.get():
                    self.send_goal(_id, _var.get())

            def on_release(_evt, _id=dxl_id, _var=goal_var):
                if self.dxl.connected:
                    self.send_goal(_id, _var.get())

            scale.configure(command=on_move)
            scale.bind("<ButtonRelease-1>", on_release)

            self.rows[dxl_id] = {
                "scale": scale,
                "goal": goal_var,
                "present": present_var,
            }

        # 연결 상태에 맞춰 disable/enable
        self._set_connected_ui(self.dxl.connected)

    # ---------- 공통 명령 ----------
    def apply_torque_all(self):
        if not self.dxl.connected:
            return
        on = 1 if self.torque_on.get() else 0
        errs = []
        for dxl_id in self.rows.keys():
            emsg = self.dxl.write1(dxl_id, ADDR_TORQUE_ENABLE, on)
            if emsg:
                errs.append(f"ID {dxl_id}: {emsg}")
        if errs:
            self.error_text.set(" | ".join(errs[:3]) + (" ..." if len(errs) > 3 else ""))

    def emergency_stop(self):
        self.torque_on.set(False)
        self.apply_torque_all()

    def apply_speed_all(self):
        if not self.dxl.connected:
            return
        speed = int(self.speed_var.get())
        errs = []
        for dxl_id in self.rows.keys():
            emsg = self.dxl.write2(dxl_id, ADDR_MOVING_SPEED, speed)
            if emsg:
                errs.append(f"ID {dxl_id}: {emsg}")
        if errs:
            self.error_text.set(" | ".join(errs[:3]) + (" ..." if len(errs) > 3 else ""))

    def apply_torque_limit_all(self):
        if not self.dxl.connected:
            return
        tl = int(self.torque_limit_var.get())
        errs = []
        for dxl_id in self.rows.keys():
            emsg = self.dxl.write2(dxl_id, ADDR_TORQUE_LIMIT, tl)
            if emsg:
                errs.append(f"ID {dxl_id}: {emsg}")
        if errs:
            self.error_text.set(" | ".join(errs[:3]) + (" ..." if len(errs) > 3 else ""))

    def send_goal(self, dxl_id: int, goal: int):
        if not self.dxl.connected:
            return
        goal = max(POS_MIN, min(POS_MAX, int(goal)))
        emsg = self.dxl.write2(dxl_id, ADDR_GOAL_POSITION, goal)
        if emsg:
            # 슬라이더 드래그 중엔 팝업은 너무 시끄러워서 하단 에러 라벨로만
            self.error_text.set(f"ID {dxl_id}: {emsg}")

    # ---------- 폴링 ----------
    def start_polling(self):
        self.stop_polling()
        self._poll_positions()

    def stop_polling(self):
        if self.poll_job is not None:
            try:
                self.after_cancel(self.poll_job)
            except Exception:
                pass
        self.poll_job = None

    def _poll_positions(self):
        if self.dxl.connected:
            for dxl_id, w in self.rows.items():
                val, emsg = self.dxl.read2(dxl_id, ADDR_PRESENT_POSITION)
                if emsg:
                    # 통신 에러가 있다면 표시만 (연결을 강제로 끊진 않음)
                    w["present"].set("present: err")
                else:
                    w["present"].set(f"present: {val}")
        else:
            for dxl_id, w in self.rows.items():
                w["present"].set("present: -")

        self.poll_job = self.after(200, self._poll_positions)

    def on_close(self):
        try:
            self.disconnect()
        finally:
            self.destroy()


if __name__ == "__main__":
    App().mainloop()
