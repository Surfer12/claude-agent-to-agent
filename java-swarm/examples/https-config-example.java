// Example: Custom HTTPS configuration for Java Swarm
// This example shows how to configure HTTPS settings, SSL, and proxy support

import com.swarm.client.StreamingOpenAIClient;
import com.swarm.core.Swarm;
import com.swarm.types.*;
import okhttp3.*;
import javax.net.ssl.*;
import java.security.cert.X509Certificate;
import java.util.*;
import okhttp3.internal.tls.OkHostnameVerifier;
import java.util.concurrent.TimeUnit;

public class HttpsConfigExample {
    
    public static void main(String[] args) {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            System.err.println("Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }
        
        // Example 1: Basic secure configuration (recommended)
        basicSecureConfiguration(apiKey);
        
        // Example 2: Custom SSL configuration
        customSslConfiguration(apiKey);
        
        // Example 3: Proxy configuration
        proxyConfiguration(apiKey);
        
        // Example 4: Custom timeouts and connection pooling
        customTimeoutsConfiguration(apiKey);
    }
    
    /**
     * Basic secure HTTPS configuration with standard settings.
     */
    private static void basicSecureConfiguration(String apiKey) {
        System.out.println("=== Basic Secure Configuration ===");
        
        // Create client with default secure settings
        StreamingOpenAIClient client = new StreamingOpenAIClient(apiKey);
        Swarm swarm = new Swarm(apiKey, client);
        
        // Test connection
        boolean connected = swarm.testConnection();
        System.out.println("Connection test: " + (connected ? "✅ Success" : "❌ Failed"));
        
        swarm.close();
        System.out.println();
    }
    
    /**
     * Custom SSL configuration for specific security requirements.
     */
    private static void customSslConfiguration(String apiKey) {
        System.out.println("=== Custom SSL Configuration ===");
        
        try {
            // Create custom SSL context (example - use proper certificates in production)
            SSLContext sslContext = SSLContext.getInstance("TLS");
            
            // Use the default trust manager for proper certificate validation
            // This ensures proper SSL/TLS security by validating certificates
            TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            trustManagerFactory.init((java.security.KeyStore) null); // Use default keystore
            TrustManager[] trustManagers = trustManagerFactory.getTrustManagers();
            
            sslContext.init(null, trustManagers, new java.security.SecureRandom());
            
            // Create custom HTTP client with SSL configuration
            OkHttpClient customHttpClient = new OkHttpClient.Builder()
                    .sslSocketFactory(sslContext.getSocketFactory(), (X509TrustManager) trustManagers[0])
                    // Use default hostname verifier for proper security
                    .hostnameVerifier(OkHostnameVerifier.INSTANCE)
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .build();
            
            StreamingOpenAIClient client = new StreamingOpenAIClient(apiKey, customHttpClient);
            Swarm swarm = new Swarm(apiKey, client);
            
            boolean connected = swarm.testConnection();
            System.out.println("Custom SSL connection test: " + (connected ? "✅ Success" : "❌ Failed"));
            
            swarm.close();
            
        } catch (Exception e) {
            System.err.println("SSL configuration error: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Proxy configuration for corporate environments.
     */
    private static void proxyConfiguration(String apiKey) {
        System.out.println("=== Proxy Configuration ===");
        
        // Example proxy settings (adjust for your environment)
        String proxyHost = System.getProperty("https.proxyHost", "proxy.company.com");
        String proxyPort = System.getProperty("https.proxyPort", "8080");
        String proxyUser = System.getProperty("https.proxyUser");
        String proxyPassword = System.getProperty("https.proxyPassword");
        
        if (proxyHost != null && !proxyHost.isEmpty()) {
            try {
                // Create proxy
                Proxy proxy = new Proxy(Proxy.Type.HTTP, 
                    new java.net.InetSocketAddress(proxyHost, Integer.parseInt(proxyPort)));
                
                OkHttpClient.Builder clientBuilder = new OkHttpClient.Builder()
                        .proxy(proxy)
                        .connectTimeout(30, TimeUnit.SECONDS)
                        .readTimeout(60, TimeUnit.SECONDS);
                
                // Add proxy authentication if needed
                if (proxyUser != null && proxyPassword != null) {
                    Authenticator proxyAuthenticator = (route, response) -> {
                        String credential = Credentials.basic(proxyUser, proxyPassword);
                        return response.request().newBuilder()
                                .header("Proxy-Authorization", credential)
                                .build();
                    };
                    clientBuilder.proxyAuthenticator(proxyAuthenticator);
                }
                
                OkHttpClient customHttpClient = clientBuilder.build();
                StreamingOpenAIClient client = new StreamingOpenAIClient(apiKey, customHttpClient);
                Swarm swarm = new Swarm(apiKey, client);
                
                System.out.println("Using proxy: " + proxyHost + ":" + proxyPort);
                boolean connected = swarm.testConnection();
                System.out.println("Proxy connection test: " + (connected ? "✅ Success" : "❌ Failed"));
                
                swarm.close();
                
            } catch (Exception e) {
                System.err.println("Proxy configuration error: " + e.getMessage());
            }
        } else {
            System.out.println("No proxy configuration found (set https.proxyHost system property)");
        }
        
        System.out.println();
    }
    
    /**
     * Custom timeouts and connection pooling configuration.
     */
    private static void customTimeoutsConfiguration(String apiKey) {
        System.out.println("=== Custom Timeouts Configuration ===");
        
        // Create HTTP client with custom timeouts and connection pooling
        ConnectionPool connectionPool = new ConnectionPool(
            20,  // maxIdleConnections
            10,  // keepAliveDuration
            TimeUnit.MINUTES
        );
        
        OkHttpClient customHttpClient = new OkHttpClient.Builder()
                .connectTimeout(45, TimeUnit.SECONDS)      // Connection timeout
                .readTimeout(120, TimeUnit.SECONDS)        // Read timeout (for long responses)
                .writeTimeout(60, TimeUnit.SECONDS)        // Write timeout
                .callTimeout(180, TimeUnit.SECONDS)        // Overall call timeout
                .connectionPool(connectionPool)            // Custom connection pool
                .retryOnConnectionFailure(true)            // Retry on connection failure
                .followRedirects(true)                     // Follow redirects
                .followSslRedirects(true)                  // Follow SSL redirects
                .addInterceptor(new LoggingInterceptor())  // Add request/response logging
                .build();
        
        StreamingOpenAIClient client = new StreamingOpenAIClient(apiKey, customHttpClient);
        Swarm swarm = new Swarm(apiKey, client);
        
        System.out.println("Custom timeouts configured:");
        System.out.println("  Connect: 45s, Read: 120s, Write: 60s, Call: 180s");
        System.out.println("  Connection pool: 20 max idle, 10min keep-alive");
        
        boolean connected = swarm.testConnection();
        System.out.println("Custom timeouts connection test: " + (connected ? "✅ Success" : "❌ Failed"));
        
        swarm.close();
        System.out.println();
    }
    
    /**
     * Custom logging interceptor for debugging HTTP requests.
     */
    private static class LoggingInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws java.io.IOException {
            Request request = chain.request();
            
            long startTime = System.currentTimeMillis();
            System.out.println("🔗 HTTP Request: " + request.method() + " " + request.url());
            
            Response response = chain.proceed(request);
            
            long endTime = System.currentTimeMillis();
            System.out.println("📡 HTTP Response: " + response.code() + " " + response.message() + 
                             " (" + (endTime - startTime) + "ms)");
            
            return response;
        }
    }
}

/*
Usage examples:

1. Basic usage:
   java -cp target/java-swarm-1.0.0.jar HttpsConfigExample

2. With proxy:
   java -Dhttps.proxyHost=proxy.company.com -Dhttps.proxyPort=8080 \
        -Dhttps.proxyUser=username -Dhttps.proxyPassword=password \
        -cp target/java-swarm-1.0.0.jar HttpsConfigExample

3. With debug logging:
   java -DDEBUG=true -cp target/java-swarm-1.0.0.jar HttpsConfigExample

Expected output:
=== Basic Secure Configuration ===
Connection test: ✅ Success

=== Custom SSL Configuration ===
Verifying hostname: api.openai.com
Validating server certificate: CN=api.openai.com
Custom SSL connection test: ✅ Success

=== Proxy Configuration ===
No proxy configuration found (set https.proxyHost system property)

=== Custom Timeouts Configuration ===
Custom timeouts configured:
  Connect: 45s, Read: 120s, Write: 60s, Call: 180s
  Connection pool: 20 max idle, 10min keep-alive
🔗 HTTP Request: GET https://api.openai.com/v1/models
📡 HTTP Response: 200 OK (234ms)
Custom timeouts connection test: ✅ Success
*/
