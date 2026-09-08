import { AccessToken } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const room = req.nextUrl.searchParams.get("room") ?? "voice-room";
  const identity = req.nextUrl.searchParams.get("identity") ?? "user";

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;

  if (!apiKey || !apiSecret) {
    return NextResponse.json(
      { error: "Missing LIVEKIT_API_KEY or LIVEKIT_API_SECRET env vars" },
      { status: 500 }
    );
  }

  const token = new AccessToken(apiKey, apiSecret, { identity });
  token.addGrant({ roomJoin: true, room, canPublish: true, canSubscribe: true });
  const jwt = await token.toJwt();

  return NextResponse.json({ token: jwt });
}
