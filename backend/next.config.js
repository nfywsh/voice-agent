/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  serverExternalPackages: [],
  // Force Next.js to listen on all interfaces
  server: {
    hostname: '0.0.0.0',
  },
}

module.exports = nextConfig