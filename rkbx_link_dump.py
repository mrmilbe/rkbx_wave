# Copyright (c) HorstHorstmann

#!/usr/bin/env python3
"""Dump every OSC message from Rekordbox Link (debug helper)."""

from pythonosc import dispatcher, osc_server

IP = "127.0.0.1"
PORT = 4460


def dump_handler(address: str, *args) -> None:
    if args:
        print(f"{address}: {args}")
    else:
        print(f"{address}: (no data)")


def main() -> None:
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(lambda addr, *a: dump_handler(addr, *a))
    print(f"Listening to ALL Rekordbox Link OSC messages on {IP}:{PORT}")
    print("Press Ctrl+C to stop.")
    server = osc_server.ThreadingOSCUDPServer((IP, PORT), disp)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dump listener...")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
