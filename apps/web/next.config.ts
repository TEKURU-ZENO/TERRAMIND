import type { NextConfig } from "next";

// The Python API (terramind serve) runs on localhost:8000 by default.
const api = process.env.TERRAMIND_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
};

export default nextConfig;
