"""Run the bot against an Android phone over ADB Wi-Fi."""

import os
import re
import socket
import shutil
import subprocess
import sys
import time

import main as bot


# The phone and the computer must be on the same network.
# Optional override. When omitted, ADB mDNS discovery finds Android's current
# wireless-debugging connect port, which can change after every restart.
PHONE_ADDRESS = os.environ.get("RUCoy_PHONE_ADDRESS")
# Optional IP-only target. Its current ADB port is still discovered automatically.
#   $env:RUCoy_PHONE_IP = "192.168.0.190"
TARGET_PHONE_IP = os.environ.get("RUCoy_PHONE_IP")
# Optional manual fallback, used only when mDNS is unavailable:
#   $env:RUCoy_FALLBACK_ADDRESS = "192.168.0.132:40029"
FALLBACK_PHONE_ADDRESS = os.environ.get("RUCoy_FALLBACK_ADDRESS")
# Android 11+ may require a separate pairing address and six-digit code.
PAIRING_ADDRESS = os.environ.get("ADB_PAIRING_ADDRESS")
PAIRING_CODE = os.environ.get("ADB_PAIRING_CODE")
ADB_PATH = os.environ.get("ADB_PATH")


def find_adb():
    """Find the Android platform-tools executable on the computer."""
    if ADB_PATH and os.path.isfile(ADB_PATH):
        return ADB_PATH

    adb = shutil.which("adb")
    if adb:
        return adb

    common_paths = (
        r"C:\Android\platform-tools\adb.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    )
    for path in common_paths:
        path = os.path.expandvars(path)
        if os.path.isfile(path):
            return path
    return None


def split_address(address):
    try:
        host, port = address.rsplit(":", 1)
        return host, int(port)
    except (AttributeError, ValueError):
        return None, None


def address_uses_ip(address, target_ip):
    host, _ = split_address(address)
    return target_ip is None or host == target_ip


def endpoint_status(address):
    """Return whether the configured TCP endpoint can be reached at all."""
    host, port = split_address(address)
    if host is None:
        return "invalid"
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return "reachable"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "unreachable"


def adb_output(adb_path, *args, timeout=10):
    return subprocess.run(
        [adb_path, *args], capture_output=True, text=True, timeout=timeout,
    )


def discover_adb_services(adb_path):
    """Find Android Wireless Debugging connect/pair ports advertised by mDNS."""
    try:
        result = adb_output(adb_path, "mdns", "services", timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None, None

    connect_addresses = []
    pairing_addresses = []
    for line in (result.stdout + result.stderr).splitlines():
        match = re.search(r"(_adb-tls-(?:connect|pairing)\._tcp)\s+(\S+)", line)
        if not match:
            continue
        service, address = match.groups()
        if service == "_adb-tls-connect._tcp":
            if address not in connect_addresses:
                connect_addresses.append(address)
        else:
            if address not in pairing_addresses:
                pairing_addresses.append(address)

    def choose_live(addresses):
        for address in addresses:
            if endpoint_status(address) == "reachable":
                return address
        return addresses[0] if addresses else None

    return choose_live(connect_addresses), choose_live(pairing_addresses)


def explain_connection_failure(address, output):
    status = endpoint_status(address)
    detail = output.strip() or "no diagnostic was returned by adb"
    print(f"[-] ADB could not connect to {address}: {detail}")
    if status == "unreachable":
        print("    The phone is not reachable from this PC. Check same Wi-Fi, phone IP, and firewall/VPN.")
    elif status == "refused":
        print("    The phone is reachable, but that port is closed or has rotated. Use the current Wireless Debugging port.")
    elif status == "reachable":
        print("    The port is open but is not answering the ADB protocol.")
        print("    Use Android Wireless debugging's 'IP address & Port', not its pairing port.")
        print("    If pairing is required, set ADB_PAIRING_ADDRESS and ADB_PAIRING_CODE, then run again.")
    else:
        print("    PHONE_ADDRESS must look like 192.168.0.190:42431.")


def init_mobile_adb():
    """Connect to the configured phone and return its ADB command details."""
    adb_path = find_adb()
    if not adb_path:
        print("[-] adb.exe was not found. Install Android SDK Platform-Tools or set ADB_PATH.")
        return None, None

    try:
        adb_output(adb_path, "start-server", timeout=10)

        discovered_phone, discovered_pairing = discover_adb_services(adb_path)
        if TARGET_PHONE_IP:
            discovered_phone = (
                discovered_phone
                if address_uses_ip(discovered_phone, TARGET_PHONE_IP)
                else None
            )
            discovered_pairing = (
                discovered_pairing
                if address_uses_ip(discovered_pairing, TARGET_PHONE_IP)
                else None
            )
        if discovered_phone and not PHONE_ADDRESS:
            print(f"[+] Discovered Android connect endpoint: {discovered_phone}")
        target_address = PHONE_ADDRESS or discovered_phone or FALLBACK_PHONE_ADDRESS
        if not target_address:
            print("[-] Android Wireless Debugging did not advertise a connect port.")
            print("    Automatic discovery is blocked on this network or Wireless debugging is off.")
            print("    On the phone, use Wireless debugging -> IP address & Port.")
            try:
                target_address = input("[?] Enter the phone connect address (IP:port), or press Enter to cancel: ").strip()
            except EOFError:
                target_address = ""
            if not target_address:
                return None, None
            print(f"[+] Using manually entered connect endpoint: {target_address}")

        pair_address = PAIRING_ADDRESS or discovered_pairing
        pairing_code = PAIRING_CODE
        if pairing_code and not pair_address:
            print("[-] ADB pairing code was supplied, but no pairing port was discovered.")
            print("    Set ADB_PAIRING_ADDRESS to the port shown by 'Pair device with pairing code'.")
            return None, None
        if pair_address:
            if not pairing_code:
                try:
                    pairing_code = input(
                        f"[?] Enter the six-digit pairing code shown on the phone "
                        f"({pair_address}): "
                    ).strip()
                except EOFError:
                    pairing_code = ""
            if not pairing_code:
                print("[-] No pairing code entered; phone connection was cancelled.")
                return None, None

            print(f"[+] Pairing with Android at {pair_address}...")
            paired = adb_output(adb_path, "pair", pair_address, pairing_code, timeout=15)
            paired_text = (paired.stdout + paired.stderr).strip()
            if paired.returncode != 0 or "successfully paired" not in paired_text.lower():
                print(f"[-] Android pairing failed: {paired_text or 'no diagnostic was returned'}")
                return None, None

            # Pairing can restart Wireless Debugging and rotate the connect
            # port. Never reuse the pre-pairing endpoint.
            if not PHONE_ADDRESS:
                for _ in range(5):
                    time.sleep(0.4)
                    refreshed_phone, _ = discover_adb_services(adb_path)
                    if refreshed_phone:
                        target_address = refreshed_phone
                        print(f"[+] Refreshed Android connect endpoint: {target_address}")
                        break

        print(f"[+] Connecting to Poco X3 Pro at {target_address}...")
        state = adb_output(adb_path, "-s", target_address, "get-state", timeout=5)
        if state.stdout.strip() != "device":
            connected = adb_output(adb_path, "connect", target_address, timeout=10)
            connect_text = connected.stdout + connected.stderr
            state = adb_output(adb_path, "-s", target_address, "get-state", timeout=5)
            if connected.returncode != 0 or state.stdout.strip() != "device":
                explain_connection_failure(target_address, connect_text)
                return None, None
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[-] ADB connection failed: {error}")
        return None, None

    print(f"[+] ADB connected to {target_address}.")
    return adb_path, target_address


def run_mobile():
    """Start only after the phone connection succeeds; never use Win32 fallback."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if not isinstance(original_stdout, bot.TimestampedOutputStream):
        sys.stdout = bot.TimestampedOutputStream(original_stdout)
    if not isinstance(original_stderr, bot.TimestampedOutputStream):
        sys.stderr = bot.TimestampedOutputStream(original_stderr)
    try:
        connection = init_mobile_adb()
        if connection == (None, None):
            return

        bot.init_adb = lambda: connection
        bot.PREFER_ADB = True
        bot.run_bot()
    finally:
        if isinstance(sys.stdout, bot.TimestampedOutputStream):
            sys.stdout.flush()
        if isinstance(sys.stderr, bot.TimestampedOutputStream):
            sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    run_mobile()
