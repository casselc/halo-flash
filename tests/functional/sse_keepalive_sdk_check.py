#!/usr/bin/env python3
"""The official OpenAI SDK reads a halo_serve stream with SSE keepalive comments.

Replays a streamed /v1/responses body captured from the server (with
--sse-keepalive comments before the first event) through the SDK's own HTTP
stack (an httpx mock transport, no socket), both as a raw event iterator and
through client.responses.stream, which reassembles the response and rejects a
malformed stream.

  HALO_SSE_DUMP=/path/ka.sse build/cpu-asan/server_tests \\
      --gtest_filter=OpenAiService.KeepaliveCommentsWhileAStreamWaitsAndClientsIgnoreThem
  python3 tests/functional/sse_keepalive_sdk_check.py /path/ka.sse

Exit status 0 when the SDK sees exactly the stream's data events and a
completed response.
"""

import sys

import httpx
from openai import OpenAI


def main():
    body = open(sys.argv[1], "rb").read()
    comments = body.count(b": keepalive\n\n")
    assert comments > 0, "the capture has no keepalive comments"
    data_events = body.count(b"\ndata: ") + body.startswith(b"data: ")

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = OpenAI(base_url="http://halo.invalid/v1", api_key="unused", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    events = list(client.responses.create(model="m", input="hi", stream=True))
    types = [e.type for e in events]
    assert len(events) == data_events, f"{len(events)} events from {data_events} data frames: {types}"
    assert types[0] == "response.created" and types[-1] == "response.completed", types
    with client.responses.stream(model="m", input="hi") as s:
        final = s.get_final_response()
    assert final.status == "completed", final.status
    print(f"ok  {comments} keepalive comments ignored; {len(events)} events; final status {final.status}; "
          f"text {final.output_text!r}")


if __name__ == "__main__":
    main()
