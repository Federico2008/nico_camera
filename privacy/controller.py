"""
Controllo privacy hardware e modalità ospiti.

LED (GPIO BCM config.PRIVACY_LED_PIN):
  - ACCESO:    monitoraggio attivo
  - SPENTO:    guest mode o sistema in pausa
  - LAMPEGGIO: interazione vocale in corso

Kill switch (GPIO BCM config.PRIVACY_BTN_PIN):
  - pressione breve (< 2s): toggle guest mode
  - pressione lunga (≥ 2s): spegne immediatamente tutto il monitoraggio

GPIO è opzionale: se non disponibile (dev machine, test) tutto degrada silenziosamente.
"""

import logging
import threading
import time
from typing import Callable

import config

logger = logging.getLogger(__name__)

try:
    from gpiozero import LED, Button
    _GPIO_OK = True
except ImportError:
    _GPIO_OK = False
    logger.debug("gpiozero non disponibile — privacy LED/button disabilitati.")


class PrivacyController:
    """
    Gestisce LED di attività, kill switch hardware e modalità ospiti.

    Uso tipico in main.py:
        privacy = PrivacyController()
        privacy.setup(on_kill=_shutdown)
        privacy.set_monitoring(True)
    """

    _LONG_PRESS_S = 2.0   # soglia pressione lunga → kill totale

    def __init__(self):
        self._guest_mode      = False
        self._monitoring      = False
        self._led             = None
        self._button          = None
        self._blink_thread    = None
        self._blink_stop      = threading.Event()
        self._press_start     = None
        self._on_kill         = None

        if _GPIO_OK:
            try:
                self._led    = LED(config.PRIVACY_LED_PIN)
                self._button = Button(config.PRIVACY_BTN_PIN, hold_time=self._LONG_PRESS_S)
            except Exception as exc:
                logger.warning("GPIO init fallito (pin errato?): %s", exc)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, on_kill: Callable | None = None) -> None:
        """
        Collega i callback hardware.
        on_kill() verrà chiamato su pressione lunga del kill switch.
        """
        self._on_kill = on_kill
        if self._button:
            self._button.when_pressed  = self._on_btn_pressed
            self._button.when_released = self._on_btn_released
            self._button.when_held     = self._on_btn_held

    # ------------------------------------------------------------------
    # LED
    # ------------------------------------------------------------------

    def led_on(self) -> None:
        self._stop_blink()
        if self._led:
            self._led.on()

    def led_off(self) -> None:
        self._stop_blink()
        if self._led:
            self._led.off()

    def led_blink(self, hz: float = 2.0) -> None:
        """Lampeggio non-bloccante a hz Hz."""
        self._stop_blink()
        if not self._led:
            return
        self._blink_stop.clear()
        period = 1.0 / hz

        def _blink():
            while not self._blink_stop.is_set():
                self._led.on()
                self._blink_stop.wait(period / 2)
                self._led.off()
                self._blink_stop.wait(period / 2)

        self._blink_thread = threading.Thread(target=_blink, daemon=True, name="led-blink")
        self._blink_thread.start()

    def _stop_blink(self) -> None:
        self._blink_stop.set()
        if self._blink_thread:
            self._blink_thread.join(timeout=1.0)
        self._blink_stop.clear()

    # ------------------------------------------------------------------
    # monitoring state
    # ------------------------------------------------------------------

    def set_monitoring(self, active: bool) -> None:
        self._monitoring = active
        self._update_led()

    def is_monitoring_allowed(self) -> bool:
        """False in guest mode o se il monitoraggio è stato spento."""
        return self._monitoring and not self._guest_mode

    # ------------------------------------------------------------------
    # guest mode
    # ------------------------------------------------------------------

    def enable_guest_mode(self) -> None:
        if self._guest_mode:
            return
        self._guest_mode = True
        self._update_led()
        logger.info("Guest mode attivata — monitoraggio sospeso.")

    def disable_guest_mode(self) -> None:
        if not self._guest_mode:
            return
        self._guest_mode = False
        self._update_led()
        logger.info("Guest mode disattivata — monitoraggio ripreso.")

    def toggle_guest_mode(self) -> bool:
        if self._guest_mode:
            self.disable_guest_mode()
        else:
            self.enable_guest_mode()
        return self._guest_mode

    @property
    def guest_mode(self) -> bool:
        return self._guest_mode

    # ------------------------------------------------------------------
    # button callbacks
    # ------------------------------------------------------------------

    def _on_btn_pressed(self) -> None:
        self._press_start = time.monotonic()

    def _on_btn_released(self) -> None:
        if self._press_start is None:
            return
        duration = time.monotonic() - self._press_start
        self._press_start = None
        if duration < self._LONG_PRESS_S:
            self.toggle_guest_mode()
            logger.info("Kill switch: pressione breve → guest_mode=%s", self._guest_mode)

    def _on_btn_held(self) -> None:
        logger.warning("Kill switch: pressione lunga → spegnimento monitoraggio.")
        self.led_off()
        self.set_monitoring(False)
        if self._on_kill:
            self._on_kill()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _update_led(self) -> None:
        if not self.is_monitoring_allowed():
            self.led_off()
        else:
            self.led_on()

    def shutdown(self) -> None:
        self.led_off()
        if self._led:
            self._led.close()
        if self._button:
            self._button.close()


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    print(f"gpiozero disponibile: {_GPIO_OK}")

    ctrl = PrivacyController()
    kill_called = []
    ctrl.setup(on_kill=lambda: kill_called.append(True))

    # ------------------------------------------------------------------
    # 1. stato iniziale
    # ------------------------------------------------------------------
    print("\n=== 1. stato iniziale ===")
    assert not ctrl.guest_mode
    assert not ctrl.is_monitoring_allowed()   # monitoring non ancora attivato
    print(f"  guest_mode={ctrl.guest_mode}  monitoring_allowed={ctrl.is_monitoring_allowed()}  OK")

    # ------------------------------------------------------------------
    # 2. set_monitoring
    # ------------------------------------------------------------------
    print("\n=== 2. set_monitoring ===")
    ctrl.set_monitoring(True)
    assert ctrl.is_monitoring_allowed()
    ctrl.set_monitoring(False)
    assert not ctrl.is_monitoring_allowed()
    ctrl.set_monitoring(True)
    print("  set_monitoring True/False/True  OK")

    # ------------------------------------------------------------------
    # 3. guest mode
    # ------------------------------------------------------------------
    print("\n=== 3. guest mode ===")
    ctrl.enable_guest_mode()
    assert ctrl.guest_mode
    assert not ctrl.is_monitoring_allowed()   # guest mode blocca monitoring
    print(f"  guest_mode attivata → monitoring_allowed={ctrl.is_monitoring_allowed()}  OK")

    ctrl.disable_guest_mode()
    assert not ctrl.guest_mode
    assert ctrl.is_monitoring_allowed()
    print(f"  guest_mode disattivata → monitoring_allowed={ctrl.is_monitoring_allowed()}  OK")

    # ------------------------------------------------------------------
    # 4. toggle
    # ------------------------------------------------------------------
    print("\n=== 4. toggle_guest_mode ===")
    states = [ctrl.toggle_guest_mode() for _ in range(4)]
    assert states == [True, False, True, False], f"atteso alternanza, got {states}"
    print(f"  toggle x4: {states}  OK")

    # ------------------------------------------------------------------
    # 5. kill switch simulato (senza GPIO hardware)
    # ------------------------------------------------------------------
    print("\n=== 5. kill switch simulato ===")
    ctrl._on_kill = lambda: kill_called.append(True)
    ctrl._on_btn_held()    # simula pressione lunga
    assert kill_called, "on_kill non chiamata"
    assert not ctrl._monitoring
    print(f"  Kill switch: on_kill chiamata={len(kill_called)} volte  OK")

    # ------------------------------------------------------------------
    # 6. LED blink senza hardware (deve non crashare)
    # ------------------------------------------------------------------
    print("\n=== 6. led_blink senza GPIO ===")
    ctrl.led_blink(hz=10)
    time.sleep(0.3)
    ctrl.led_off()
    print("  Nessun crash senza GPIO  OK")

    ctrl.shutdown()
    print("\n=== Tutti i test superati ===")
