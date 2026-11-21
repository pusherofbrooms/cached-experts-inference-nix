"""Expose the serving subset of Strata Terminal, not its installer/launcher."""
import sys


def main(args):
    if not args or args in (['--help'], ['-h']):
        print('Usage: strata-terminal {serve,status} [options]\n'
              '  serve   Plan or run a foreground loopback server from a local config\n'
              '  status  Read a telemetry snapshot from a running server\n'
              'Run strata-terminal serve --help or strata-terminal status --help.')
        return 0
    if args[0] not in ('serve', 'status') and args != ['--version']:
        print('strata-terminal: only serve, status and --version are packaged', file=sys.stderr)
        return 2

    # Call dispatch directly: strata.py also hands over to user installs and
    # automatically migrates existing run configs before dispatch.
    from strata_cli.cli import main as dispatch
    if args[0] == 'serve':
        # Last option wins, including over profile/config values. No automatic
        # acquisition, remote image fetches, or MCP for this entry point.
        args = [*args, '--host=127.0.0.1', '--offline']
    return dispatch(args)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
