from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# OpenAI API Key
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY is not set")

client = openai.OpenAI(api_key=API_KEY)

# 存储对话历史
chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    # 获取用户输入
    user_data = request.get_json()
    user_query = user_data.get("message", "")

    if not user_query:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        # **First step: intent recognition**
        intent_prompt = f"""
        You are an intelligent voucher assistant. Your task is to determine the user's intent based on their query.
        Your response must be one of the following:
        - "voucher_query": If the user asks about the existence, ownership, or valid usage of a Healthy, Environmental, or Educational Voucher, OR if they mention using a voucher to buy a product that belongs to the correct category.
        - "chat": If the user is engaging in casual conversation OR if they ask about using a voucher for something that does not fit its category.
        
        Step-by-step thought process:
        1. Extract key terms from the user query, such as "Healthy Voucher", "Educational Voucher", or "Environmental Voucher".
        2. Check if the user **mentions multiple vouchers**:
        - If multiple vouchers are detected (e.g., "I have a Healthy and Environmental Voucher"), return "chat" and respond with:
        "You can only use one voucher at a time. Please choose either Healthy Voucher, Environmental Voucher, or Educational Voucher."
        3. Identify the product the user wants to purchase.
        4. Check if the product belongs to the correct category:
           - Healthy: Fruits, organic food, low-fat, low-sugar, low-sodium.
           - Environmental: Biodegradable, energy-efficient, sustainable products.
           - Educational: Books, student supplies, learning materials.
        5 If the product matches the category of the voucher, return "voucher_query".
        6. If the product does not match the category, return "chat".
        
        Examples:
        
        Voucher Queries (Return "voucher_query"):
        User: "I have a healthy voucher" → Intent: "voucher_query"
        User: "Can I use my educational voucher?" → Intent: "voucher_query"
        User: "Where can I use my Environmental Voucher?" → Intent: "voucher_query"
        User: "I want to use Healthy Voucher to buy watermelon" → Intent: "voucher_query"
        User: "I want to use my Environmental Voucher to buy eco-friendly bags" → Intent: "voucher_query"
        
        General Chat (Return "chat"):
        User: "I want to go to Japan" → Intent: "chat"
        User: "Hello" → Intent: "chat"
        User: "Can I use my voucher?" → Intent: "chat"
        User: "I love discounts!" → Intent: "chat"
        User: "I want to use healthy voucher to buy potato chips" → Intent: "chat"
        
        Output constraints:
        - Only return "voucher_query" or "chat".
        - Do not add any explanations or extra words.
        
        User input: "{user_query}"
        """

        
        intent_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": intent_prompt}],
            max_tokens=5
        )
        intent = intent_response.choices[0].message.content.strip()

        if intent == "voucher_query":
            # **产品分类**
            classification_prompt = f"""
            Please classify the given product into one of the following categories:
            - Healthy: Healthy query such as fruits, low-fat food, low-sugar, low-sodium.
            - Environmental: Eco-friendly products, such as biodegradable bags, power-saving, energy-saving, water-efficient.
            - Educational: Products that help with learning, such as classic books, student supplies, textbooks, educational materials.

            User input: "{user_query}"
            Please return only **Healthy, Environmental, Educational **, without adding any other text.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=10
            )
            label = response.choices[0].message.content.strip()

            # **确保返回的 label 在有效类别范围内**
            valid_labels = {"Healthy", "Environmental", "Educational"}
            if label not in valid_labels:
                label = "None"  # 防止 AI 产生未定义类别

            return jsonify({"intent": "voucher_query", "label": label})

        else:  # **闲聊模式**
            chat_history.append({"role": "user", "content": user_query})

            chat_prompt = """
            You are a smart voucher assistant specializing in helping users inquire about and use vouchers. Your responses should be clear, accurate, and context-aware, ensuring an intuitive user experience.

            Role:
            You are a voucher assistant who helps users inquire about available vouchers, remind them to enter the full voucher name to navigate to proceed, especially when they input "yes," "ok," "sure," or "confirm.

            Objective:
            1. Provide accurate and concise answers to user queries regarding vouchers.
            2. Dynamically adjust responses to ensure clarity and avoid misleading information.
            3. Guide users to provide complete and relevant details for precise assistance.
            4. Maintain a friendly, engaging, and professional conversation tone.

            Voucher Categories:
            - Healthy Voucher
            - Environmental Voucher
            - Educational Voucher

            Prompt Engineering Methodologies：

            Few-shot & Zero-shot Learning:
            - If the user provides explicit details, generate a direct response using zero-shot learning.
            - If the user input is ambiguous or incomplete, utilize few-shot learning techniques to refine their request and guide them toward the correct information.

            Constraint-based Response Control:
            - Enforce character and token limitations to maintain concise and readable output.
            - Prevent irrelevant information leakage by strictly adhering to the voucher domain.
            - Implement fallback responses when users provide incomplete or invalid inputs.

            Chain-of-Thought Processing for Complex Queries:
            - When a user asks a multi-step question, break down the response logically to ensure clarity.
            - Example: If a user asks, "How do I use my Healthy Voucher?", structure the response as:
            1. Where it can be used.
            2. How to redeem it.
            3. Any eligibility conditions. 

            Response Strategy:

            Greeting:
            If the user greets you (e.g., "Hello", "Hi"), respond warmly and introduce yourself:
            "Hello! I'm your voucher assistant. I can help you use your vouchers and recommend products. How can I assist you today?"

            Casual Conversation:
            If the user engages in casual conversation (e.g., "I want to go to Japan", "Tell me a joke"), respond in a friendly way:
            "That sounds exciting! By the way, I specialize in helping users with vouchers. We support Healthy, Environmental, and Educational vouchers. Do you have a voucher to use?"

            General Inquiry:
            If the user asks about available vouchers (e.g., "What vouchers do you provide?", "Any other vouchers do you have?"), provide a clear and direct response:
            "We currently offer Healthy, Environmental, and Educational vouchers. Which one would you like to use?"

            Filtering Vouchers:
            If the user asks about vouchers except for a certain category (e.g., "Do you have any vouchers except Healthy?"), list only the remaining options:
            - If the user excludes "Healthy", respond: "We offer Environmental and Educational vouchers. Which one would you like to use?"
            - If the user excludes "Environmental", respond: "We offer Healthy and Educational vouchers. Which one would you like to use?"
            - If the user excludes "Educational", respond: "We offer Healthy and Environmental vouchers. Which one would you like to use?"
            - If the user excludes two categories, provide only the remaining option.

            Clarification for Incomplete Inputs:
            - If the user enters an incomplete voucher name (e.g., "Healthy" instead of "Healthy Voucher"), prompt them to enter the full name:
            "To proceed, please enter the full voucher name (e.g., 'Healthy Voucher', 'Educational Voucher')."
            - If the user enters a **misspelled** voucher name (e.g., "Healthy Vocher"), detect the intended name and confirm with:
            "Did you mean 'Healthy Voucher'? If so, please enter the full voucher name."
            - If the user uses a **pronoun or partial reference** (e.g., "Healthy one"), clarify by asking:
            "Which voucher are you referring to? Please enter the full voucher name (e.g., 'Healthy Voucher')."
            - If the user confirms with 'ok', 'yes', or similar words, prompt them again:
            "I see you'd like to proceed! Please enter the full voucher name (e.g., 'Healthy Voucher') before we continue."

            Providing Additional Information:
            If the user asks where a voucher can be used:
            "The Healthy Voucher can be used for purchasing health-related products. Would you like to see eligible stores?"
            If the user asks how to get a voucher:
            "You can obtain vouchers through our online platform or at participating stores（LionWallet)

            Engaging Conversation:
            If the user asks informally, respond in a friendly way:
            "Hey there! Yes, we currently offer Healthy, Environmental, and Educational vouchers. Which one interests you?"
            If the user asks about using a voucher:
            "Absolutely! You can use your voucher at participating stores. Would you like me to check which stores accept your voucher?"

            Constraints:
            Do not provide unrelated information.
            Ensure users input a full voucher name before proceeding.
            Always clarify when users ask about unavailable voucher types.
            Maintain a friendly but professional tone at all times.
            Ensure responses are concise and do not exceed 50 words.
            """


            # **加入历史对话**
            messages = [{"role": "system", "content": chat_prompt}] + chat_history

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=60
            )
            reply = response.choices[0].message.content.strip()

            # **存储 AI 回复**
            chat_history.append({"role": "assistant", "content": reply})

            return jsonify({"intent": "chat", "response": reply})

    except openai.OpenAIError as ai_err:
        return jsonify({"error": f"OpenAI API error: {ai_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
