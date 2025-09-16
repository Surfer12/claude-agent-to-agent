/** @type {import('next').NextConfig} */
const nextConfig = {
	reactStrictMode: true,
	images: {
		// No remote domains permitted. Only local and data URLs used in app.
		domains: [],
		remotePatterns: [],
		formats: ["image/avif", "image/webp"],
		dangerouslyAllowSVG: false,
		contentDispositionType: "inline",
		contentSecurityPolicy:
			"default-src 'self'; script-src 'none'; sandbox;",
	},
};

export default nextConfig;
