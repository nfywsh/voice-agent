import { AccessToken, VideoGrant, AgentDispatchClient } from 'livekit-server-sdk';
import { NextRequest, NextResponse } from 'next/server';

/**
 * Token 生成 API
 *
 * GET /api/token?room={roomName}&username={userName}
 *
 * 生成 LiveKit 访问令牌 (JWT)，供前端连接 LiveKit 房间使用。
 * API Key 和 Secret 通过环境变量注入，严禁硬编码。
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const roomName = searchParams.get('room') || 'voice-demo';
  const userName = searchParams.get('username') || 'user';

  // 验证环境变量
  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;

  if (!apiKey || !apiSecret) {
    console.error('[Token API] LIVEKIT_API_KEY or LIVEKIT_API_SECRET not configured');
    return NextResponse.json(
      { error: 'LiveKit credentials not configured on server' },
      { status: 500 }
    );
  }

  // 验证参数
  if (!roomName.trim() || !userName.trim()) {
    return NextResponse.json(
      { error: 'room and username are required' },
      { status: 400 }
    );
  }

  // 限制房间名和用户名长度
  const sanitizedRoom = roomName.trim().slice(0, 64);
  const sanitizedUser = userName.trim().slice(0, 64);

  try {
    // 创建访问令牌
    const token = new AccessToken(apiKey, apiSecret, {
      identity: sanitizedUser,
      name: sanitizedUser,
      // 令牌有效期 1 小时
      ttl: '1h',
    });

    // 授予房间权限
    const grant: VideoGrant = {
      room: sanitizedRoom,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    };
    token.addGrant(grant);

    const jwt = await token.toJwt();

    // 显式调度 Agent 入房（绕过 auto-dispatch 机制）
    // auto-dispatch（token 中 agent:true）在当前环境无法正常工作，
    // 因此通过 API 显式创建 dispatch，让 Worker 接收 Job 并入房
    try {
      const livekitHost = process.env.LIVEKIT_HOST || 'http://livekit:7880';
      const dispatchClient = new AgentDispatchClient(livekitHost, apiKey, apiSecret);

      // 清理该房间的旧 dispatch，确保不会有 stale dispatch 导致 agent 不响应
      // 问题：刷新页面或离开重进时，旧 dispatch 可能处于僵死状态（对应 worker 已退出）
      // 解决方案：直接 list 并删除所有旧 dispatch，不依赖 listDispatch 的成功
      try {
        const existingDispatches = await dispatchClient.listDispatch(sanitizedRoom);
        if (existingDispatches && existingDispatches.length > 0) {
          for (const dispatch of existingDispatches) {
            try {
              await dispatchClient.deleteDispatch(dispatch.id, sanitizedRoom);
              console.log(`[Token API] Deleted stale dispatch ${dispatch.id} for room: ${sanitizedRoom}`);
            } catch (deleteErr) {
              console.warn(`[Token API] Failed to delete dispatch ${dispatch.id}:`, deleteErr);
            }
          }
        }
      } catch (listErr) {
        // listDispatch 可能失败（如 agent worker 不稳定），此时直接创建新 dispatch
        // LiveKit 会自动处理旧 dispatch（超时后自动清理）
        console.warn(`[Token API] listDispatch failed, proceeding with createDispatch:`, listErr);
      }

      // 创建新 dispatch
      const newDispatch = await dispatchClient.createDispatch(sanitizedRoom, '');
      console.log(`[Token API] Agent dispatch created: id=${newDispatch.id} for room: ${sanitizedRoom}`);
    } catch (dispatchError) {
      // dispatch 失败不影响 token 生成，仅记录警告
      console.warn(`[Token API] Failed to create agent dispatch for room ${sanitizedRoom}:`, dispatchError);
    }

    return NextResponse.json(
      { token: jwt, room: sanitizedRoom, identity: sanitizedUser },
      {
        headers: {
          'Cache-Control': 'no-store',  // 令牌不应被缓存
        },
      }
    );
  } catch (error) {
    console.error('[Token API] Error generating token:', error);
    return NextResponse.json(
      { error: 'Failed to generate token' },
      { status: 500 }
    );
  }
}