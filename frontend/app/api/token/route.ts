import { AccessToken } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const ROOM_REGEX = /^[a-zA-Z0-9_-]{1,64}$/;
const IDENTITY_REGEX = /^[a-zA-Z0-9_-]{1,128}$/;

export async function GET(req: NextRequest) {
  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;

  if (!apiKey || !apiSecret) {
    console.error("Missing LiveKit API credentials in server environment");
    return NextResponse.json(
      { error: "Server configuration error" },
      { status: 500 }
    );
  }

  // 1. Sanitize parameters & reject empty string injection
  const rawRoom = req.nextUrl.searchParams.get("room")?.trim();
  const rawIdentity = req.nextUrl.searchParams.get("identity")?.trim();

  const room = rawRoom && rawRoom.length > 0 ? rawRoom : "voice-room";
  const identity =
    rawIdentity && rawIdentity.length > 0
      ? rawIdentity
      : `user-${Math.random().toString(36).substring(2, 9)}`;

  // 2. Validate formats strictly
  if (!ROOM_REGEX.test(room)) {
    return NextResponse.json(
      { error: "Invalid room format. Only alphanumeric characters, dashes, and underscores allowed (max 64)." },
      { status: 400 }
    );
  }

  if (!IDENTITY_REGEX.test(identity)) {
    return NextResponse.json(
      { error: "Invalid identity format. Only alphanumeric characters, dashes, and underscores allowed (max 128)." },
      { status: 400 }
    );
  }

  try {
    // 3. Mint restricted token with 10-minute TTL
    const token = new AccessToken(apiKey, apiSecret, {
      identity,
      ttl: "10m",
    });

    token.addGrant({
      roomJoin: true,
      room,
      canPublish: true,
      canPublishSources: ["microphone"], // Restrict track publication to mic only
      canSubscribe: true,
      canPublishData: true,
    });

    const jwt = await token.toJwt();

    return NextResponse.json(
      { token: jwt },
      {
        headers: {
          "Cache-Control": "no-store, max-age=0",
        },
      }
    );
  } catch (err) {
    console.error("Token generation failed:", err);
    return NextResponse.json(
      { error: "Internal server error generating room credentials" },
      { status: 500 }
    );
  }
}