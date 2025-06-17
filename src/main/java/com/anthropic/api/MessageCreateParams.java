import com.anthropic.models.Message;
import com.anthropic.models.MessageCreateParams;
import com.anthropic.models.Model;

MessageCreateParams params = MessageCreateParams.builder()
    .maxTokens(1024L)
    .addUserMessage("Hello, Claude")
    .model(Model.CLAUDE_SONNET_4_0)
    .build();
Message message = client.messages().create(params);