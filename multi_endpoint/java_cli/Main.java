package multi_endpoint.java_cli;

public class Main {
    public static void main(String[] args) {
        String apiKey = System.getenv().getOrDefault("MOCK_API_KEY", "mock-api-key");
        if (args.length == 0) {
            System.out.println("Please provide a message to echo");
            return;
        }
        System.out.println("Java endpoint using API key: " + apiKey);
        System.out.println("Message: " + args[0]);
    }
}
