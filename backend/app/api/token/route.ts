import { AccessToken, VideoGrant, AgentDispatchClient } from 'livekit-server-sdk';
import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

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
  const refAudioPath = searchParams.get('ref_audio_path') || '';
  const refText = searchParams.get('ref_text') || '';

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

  // 保存参考音色参数到 session（异步，不阻塞 token 返回）
  if (refAudioPath) {
    try {
      const { saveSessionParams } = await import('../session/service');
      saveSessionParams(sanitizedRoom, {
        ref_audio_path: refAudioPath,
        ref_text: refText,
      }).catch(console.error);
    } catch {
      /* ignore */
    }
  }

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
    try {
      const livekitHost = process.env.LIVEKIT_HOST || 'http://livekit:7880';
      const dispatchClient = new AgentDispatchClient(livekitHost, apiKey, apiSecret);

      // 并发控制：同一房间同时只有一个请求执行 dispatch 清理+创建
      const lockFilePath = `/tmp/dispatch-lock-${sanitizeFilename(sanitizedRoom)}`;
      let lockAcquired = false;
      try {
        // 使用 mkdir 原子操作获取锁（POSIX mkdir 是原子的）
        await fs.mkdir(lockFilePath, { recursive: false });
        lockAcquired = true;
      } catch (mkdirErr: any) {
        if (mkdirErr.code === 'EEXIST') {
          console.warn(`[Token API] Another request is handling dispatch for room ${sanitizedRoom}, waiting...`);
          // 等待锁释放（通过轮询检查目录是否消失）
          for (let i = 0; i < 20; i++) {
            await new Promise(r => setTimeout(r, 500));
            try {
              await fs.mkdir(lockFilePath, { recursive: false });
              lockAcquired = true;
              break;
            } catch {
              continue;
            }
          }
          if (!lockAcquired) {
            console.warn(`[Token API] Timeout waiting for lock, proceeding anyway`);
          }
        } else {
          throw mkdirErr;
        }
      }

      if (!lockAcquired) {
        // 无法获取锁，直接返回 token（让请求继续，dispatch 创建可能失败但不影响用户体验）
        return NextResponse.json(
          { token: jwt, room: sanitizedRoom, identity: sanitizedUser },
          { headers: { 'Cache-Control': 'no-store' } }
        );
      }

      try {
        // 尝试清理该房间的旧 dispatch（listDispatch 失败时也尝试用已知 dispatch ID 清理）
        // 方法1: listDispatch 获取并清理
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
          console.warn(`[Token API] listDispatch failed:`, listErr);
          // listDispatch 失败时，尝试读取本地缓存的 dispatch ID 进行清理
          await cleanupKnownDispatch(dispatchClient, sanitizedRoom);
        }

        const newDispatch = await dispatchClient.createDispatch(sanitizedRoom, '');
        console.log(`[Token API] Agent dispatch created: id=${newDispatch.id} for room: ${sanitizedRoom}`);

        // 保存 dispatch ID 到本地缓存，供下次进入房间时清理
        await saveKnownDispatch(sanitizedRoom, newDispatch.id);
      } finally {
        // 释放锁
        try {
          await fs.rmdir(lockFilePath);
        } catch {
          /* ignore */
        }
      }
    } catch (dispatchError) {
      console.warn(`[Token API] Failed to create agent dispatch for room ${sanitizedRoom}:`, dispatchError);
    }

    return NextResponse.json(
      { token: jwt, room: sanitizedRoom, identity: sanitizedUser },
      {
        headers: {
          'Cache-Control': 'no-store',
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

// ---------------------------------------------------------------------------
// Dispatch ID 本地缓存：解决 listDispatch 失败时无法清理残留 dispatch 的问题
// ---------------------------------------------------------------------------

const DISPATCH_CACHE_DIR = '/tmp/dispatch-cache';

/** 从本地缓存读取该房间上次创建的 dispatch ID */
async function getKnownDispatchId(roomName: string): Promise<string | null> {
  try {
    const filePath = path.join(DISPATCH_CACHE_DIR, `${sanitizeFilename(roomName)}.json`);
    const data = await fs.readFile(filePath, 'utf-8');
    const { dispatchId } = JSON.parse(data);
    return dispatchId || null;
  } catch {
    return null;
  }
}

/** 保存 dispatch ID 到本地缓存 */
async function saveKnownDispatch(roomName: string, dispatchId: string): Promise<void> {
  try {
    await fs.mkdir(DISPATCH_CACHE_DIR, { recursive: true });
    const filePath = path.join(DISPATCH_CACHE_DIR, `${sanitizeFilename(roomName)}.json`);
    await fs.writeFile(filePath, JSON.stringify({ dispatchId, updatedAt: new Date().toISOString() }), 'utf-8');
  } catch (e) {
    console.warn(`[Token API] Failed to save dispatch cache for ${roomName}:`, e);
  }
}

/** 删除本地缓存的 dispatch 记录 */
async function deleteKnownDispatch(roomName: string): Promise<void> {
  try {
    const filePath = path.join(DISPATCH_CACHE_DIR, `${sanitizeFilename(roomName)}.json`);
    await fs.unlink(filePath);
  } catch {
    /* ignore */
  }
}

/**
 * listDispatch 失败时，尝试用本地缓存的 dispatch ID 进行清理。
 * 这样即使 LiveKit API 不可用，也能清理上次留下的旧 dispatch。
 */
async function cleanupKnownDispatch(dispatchClient: AgentDispatchClient, roomName: string): Promise<void> {
  const knownId = await getKnownDispatchId(roomName);
  if (!knownId) {
    console.warn(`[Token API] No known dispatch ID to clean up for room: ${roomName}`);
    return;
  }
  try {
    await dispatchClient.deleteDispatch(knownId, roomName);
    console.log(`[Token API] Deleted known stale dispatch ${knownId} for room: ${roomName}`);
  } catch (e) {
    console.warn(`[Token API] Failed to delete known dispatch ${knownId}:`, e);
    // dispatch 可能已经自动清理或不存在，忽略错误
  } finally {
    await deleteKnownDispatch(roomName);
  }
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, '_');
}