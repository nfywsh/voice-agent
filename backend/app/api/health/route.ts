import { NextResponse } from 'next/server';

/**
 * Health Check API
 *
 * GET /api/health
 *
 * 用于 Docker HEALTHCHECK 和负载均衡器探针。
 */
export async function GET() {
  return NextResponse.json({ status: 'ok' });
}