import argparse

from .ui.server import DEFAULT_HOST, DEFAULT_PORT

parser = argparse.ArgumentParser(
    prog="frontdesk", description="Voice front-end for Mystin Office."
)
parser.add_argument(
    "--ui", action="store_true", help="run the browser interface instead of the terminal"
)
parser.add_argument("--host", default=DEFAULT_HOST, help=f"UI bind address (default {DEFAULT_HOST})")
parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UI port (default {DEFAULT_PORT})")
parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
args = parser.parse_args()

if args.ui:
    from .ui import run

    run(host=args.host, port=args.port, open_browser=not args.no_browser)
else:
    from .core import main

    main()
