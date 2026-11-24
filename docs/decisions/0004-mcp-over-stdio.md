# 0004. MCP over stdio for serving the model

Status: Accepted

## Context

The course ends with another AI app, such as Claude Code, calling the reader's own model. MCP (Model Context Protocol) is a standard for that: a client lists a server's tools and calls them. The server library (`mcp`, `MCPServer`) supports three transports: `stdio`, `sse` and `streamable-http` (signature of `MCPServer.run`).

The reader's model runs on their own laptop and answers a tool called `consult_homer_llm` with a seed and a length.

## Decision

The server runs over `stdio` and nothing else. `homer_mcp_server.main()` calls `mcp.run()` with the default transport. The client starts it as a child process (`uv run homer-serve`) and talks over stdin and stdout.

Because stdout is the protocol channel, everything else (startup line, one log line per call) is printed to stderr.

## Consequences

- Nothing to configure: no port, no URL, no certificate, no token. The server exists only while the client is running, and nothing else on the machine can connect to it.
- No network exposure to reason about. The server code makes no network calls (see the security notes in [ARCHITECTURE.md](../ARCHITECTURE.md)).
- A stray `print` to stdout would corrupt the protocol. Logging goes to stderr by rule, and `tests/test_mcp_server.py` tests the server file.
- One client per process, and the model loads once per process (`load()` caches it). Each client that starts the server pays the load time.
- The tool input is clamped (1 to 200 characters), but there is no authentication or rate limit, because only the local client can reach the process.
- Not possible: calling the model from another machine or a browser, or sharing one running model between several users.
- What streamable HTTP would add: a long-running shared server, remote clients, and several users on one loaded model. It would also need authentication, TLS, an origin check, request limits and a decision about who may call it. That's a different project from a beginner course on a laptop. Revisit if the course wants a hosted demo; the tool functions (`find_lines`, `clamp_chars`, `next_scores`) do not depend on the transport, so only `main()` and deployment would change.
