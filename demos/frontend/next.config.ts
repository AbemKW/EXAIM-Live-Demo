import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: false, // Disabled to prevent duplicate WebSocket connections during development
};

export default nextConfig;
