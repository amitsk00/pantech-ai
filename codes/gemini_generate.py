import openai
import os

GEMINI_KEY = "xx"
os.environ['GOOGLE_API_KEY'] = GEMINI_KEY 

# Set up OpenAI API key (replace with your own API key)
# openai.api_key = ''

print("1")

try:
    client = openai.OpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

except openai.AuthenticationError:
    print("❌ Authentication Error: Your API key is likely invalid.")
except openai.APIConnectionError:
    print("❌ Connection Error: Could not reach the server. Check your internet or base_url.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

print("2")

# Function to classify text using GPT-3 zero-shot
def gpt3_zero_shot_classification(prompt):
    try:
        response = client.completions.create(
            model="gemini-2.5-flash-lite",
            prompt=prompt,
            max_tokens=50,
            temperature=0.3
        )
        return response['choices'][0]['text'].strip()

    except openai.APIConnectionError:
        print("❌ Connection Error: Could not reach the server. Check your internet or base_url.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")




# Example of zero-shot classification
text_to_classify = "The food at the restaurant was amazing!"
prompt = f"Classify the sentiment of the following review as positive, negative, or neutral:\n\n{text_to_classify}"
output = gpt3_zero_shot_classification(prompt)
print("GPT-3's Classification:", output)

