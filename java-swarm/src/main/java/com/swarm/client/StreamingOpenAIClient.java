package com.swarm.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.swarm.types.StreamingResponse;
import com.swarm.util.SwarmUtil;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.subjects.PublishSubject;
import okhttp3.*;
import okhttp3.sse.EventSource;
import okhttp3.sse.EventSourceListener;
import okhttp3.sse.EventSources;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.security.cert.X509Certificate;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * HTTPS client for OpenAI API with streaming support.
 */
public class StreamingOpenAIClient {
    private static final Logger logger = LoggerFactory.getLogger(StreamingOpenAIClient.class);
    private static final String OPENAI_API_BASE = "https://api.openai.com/v1";
    private static final String CHAT_COMPLETIONS_ENDPOINT = "/chat/completions";
    
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String apiKey;
    
    public StreamingOpenAIClient(String apiKey) {
        this.apiKey = apiKey;
        this.objectMapper = new ObjectMapper();
        this.httpClient = createSecureHttpClient();
    }
    
    public StreamingOpenAIClient(String apiKey, OkHttpClient customClient) {
        this.apiKey = apiKey;
        this.objectMapper = new ObjectMapper();
        this.httpClient = customClient;
    }
    
    /**
     * Create a secure HTTP client with proper SSL/TLS configuration.
     */
    private OkHttpClient createSecureHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .callTimeout(120, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .followRedirects(true)
                .followSslRedirects(true)
                // Add connection pooling for better performance
                .connectionPool(new ConnectionPool(10, 5, TimeUnit.MINUTES))
                // Add interceptor for logging (optional)
                .addInterceptor(new LoggingInterceptor())
                .build();
    }
    
    /**
     * Create a non-streaming chat completion request.
     */
    public String createChatCompletion(Map<String, Object> requestBody) throws IOException {
        requestBody.put("stream", false);
        
        String jsonBody = objectMapper.writeValueAsString(requestBody);
        
        RequestBody body = RequestBody.create(
                jsonBody, 
                MediaType.get("application/json; charset=utf-8")
        );
        
        Request request = new Request.Builder()
                .url(OPENAI_API_BASE + CHAT_COMPLETIONS_ENDPOINT)
                .post(body)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .addHeader("User-Agent", "Java-Swarm/1.0.0")
                .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("HTTP " + response.code() + ": " + errorBody);
            }
            
            return response.body() != null ? response.body().string() : "";
        }
    }
    
    /**
     * Create a streaming chat completion request.
     */
    public Observable<StreamingResponse> createStreamingChatCompletion(Map<String, Object> requestBody) {
        requestBody.put("stream", true);
        
        PublishSubject<StreamingResponse> subject = PublishSubject.create();
        
        try {
            String jsonBody = objectMapper.writeValueAsString(requestBody);
            
            RequestBody body = RequestBody.create(
                    jsonBody, 
                    MediaType.get("application/json; charset=utf-8")
            );
            
            Request request = new Request.Builder()
                    .url(OPENAI_API_BASE + CHAT_COMPLETIONS_ENDPOINT)
                    .post(body)
                    .addHeader("Authorization", "Bearer " + apiKey)
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Accept", "text/event-stream")
                    .addHeader("Cache-Control", "no-cache")
                    .addHeader("User-Agent", "Java-Swarm/1.0.0")
                    .build();
            
            EventSourceListener listener = new EventSourceListener() {
                @Override
                public void onOpen(EventSource eventSource, Response response) {
                    logger.debug("Streaming connection opened");
                }
                
                @Override
                public void onEvent(EventSource eventSource, String id, String type, String data) {
                    if ("[DONE]".equals(data.trim())) {
                        subject.onComplete();
                        return;
                    }
                    
                    try {
                        StreamingResponse streamingResponse = objectMapper.readValue(data, StreamingResponse.class);
                        subject.onNext(streamingResponse);
                    } catch (Exception e) {
                        logger.warn("Failed to parse streaming response: " + data, e);
                    }
                }
                
                @Override
                public void onClosed(EventSource eventSource) {
                    logger.debug("Streaming connection closed");
                    subject.onComplete();
                }
                
                @Override
                public void onFailure(EventSource eventSource, Throwable t, Response response) {
                    logger.error("Streaming connection failed", t);
                    subject.onError(t);
                }
            };
            
            EventSource eventSource = EventSources.createFactory(httpClient)
                    .newEventSource(request, listener);
            
            // Handle disposal
            subject.doOnDispose(() -> {
                eventSource.cancel();
                logger.debug("Streaming connection disposed");
            });
            
        } catch (Exception e) {
            subject.onError(e);
        }
        
        return subject;
    }
    
    /**
     * Test the connection to OpenAI API.
     */
    public boolean testConnection() {
        try {
            Request request = new Request.Builder()
                    .url(OPENAI_API_BASE + "/models")
                    .get()
                    .addHeader("Authorization", "Bearer " + apiKey)
                    .addHeader("User-Agent", "Java-Swarm/1.0.0")
                    .build();
            
            try (Response response = httpClient.newCall(request).execute()) {
                return response.isSuccessful();
            }
        } catch (Exception e) {
            logger.warn("Connection test failed", e);
            return false;
        }
    }
    
    /**
     * Get available models from OpenAI API.
     */
    public String getModels() throws IOException {
        Request request = new Request.Builder()
                .url(OPENAI_API_BASE + "/models")
                .get()
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("User-Agent", "Java-Swarm/1.0.0")
                .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("HTTP " + response.code() + ": " + response.message());
            }
            
            return response.body() != null ? response.body().string() : "";
        }
    }
    
    /**
     * Close the HTTP client and release resources.
     */
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }
    
    /**
     * Logging interceptor for debugging HTTP requests.
     */
    private static class LoggingInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            
            boolean debug = System.getenv("DEBUG") != null;
            if (debug) {
                SwarmUtil.debugPrint(true, "HTTP Request: " + request.method() + " " + request.url());
            }
            
            long startTime = System.currentTimeMillis();
            Response response = chain.proceed(request);
            long endTime = System.currentTimeMillis();
            
            if (debug) {
                SwarmUtil.debugPrint(true, "HTTP Response: " + response.code() + " in " + (endTime - startTime) + "ms");
            }
            
            return response;
        }
    }
}
