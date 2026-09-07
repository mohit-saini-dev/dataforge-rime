import { AccessToken } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/**
 * Mints a short-lived LiveKit join token for the browser client.
 *
 * The API key/secret never leave the server — the client only ever sees the
 * signed token this returns. Room name and participant identity are read
 * from the query string so the same route works for any room the backend
 * agent worker is dispatched into.
 */
export async function GET(req: NextRequest) {
  const { LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET } = process.env;

  if (!LIVEKIT_URL || !LIVEKIT_API_KEY || !LIVEKIT_API_SECRET) {
    return NextResponse.json(
      {
        error:
          "Server is missing LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET. Set them in frontend/.env.local.",
      },
      { status: 500 },
    );
  }

  const searchParams = req.nextUrl.searchParams;
  const room =
    searchParams.get("room") ??
    process.env.NEXT_PUBLIC_DEFAULT_ROOM_NAME ??
    "dataforge-rime-dev";

  // A fresh, unique identity per session avoids collisions when the same
  // browser reconnects (e.g. after a refresh) while a prior session is
  // still draining on the server.
  const identity = searchParams.get("identity") ?? `user-${crypto.randomUUID().slice(0, 8)}`;

  const at = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, {
    identity,
    ttl: "10m",
  });

  at.addGrant({
    room,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  });

  const token = await at.toJwt();

  return NextResponse.json({
    token,
    url: LIVEKIT_URL,
    room,
    identity,
  });
}
