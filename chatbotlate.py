from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# OpenAI API Key
client = openai.OpenAI(api_key="OPENAI_API_KEY")

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
        # **第一步：意图识别**
        intent_prompt = f"""
        You are an intelligent voucher assistant. Your task is to determine the user's intent and return one of the following:
        - "voucher_query": If the user mentions they HAVE, OWN, ASK about healthy, environmental, educational voucher
        - "chat": If the user is engaging in casual conversation，If the user asks about vouchers in general (e.g., "Can I use my voucher?", "I have an electronic voucher?"

        Examples:
        User: "I have a healthy voucher" → Intent: **"voucher_query"**
        User: "I own an environmental voucher" → Intent: **"voucher_query"**
        User: "Can I use my educational voucher?" → Intent: **"voucher_query"**

        User: "I want to go to Japan" → Intent: "chat"
        User: "Hello" → Intent: "chat"
        User: "How's the weather today?" → Intent: "chat"
        User: "Can I use my voucher?" → Intent: "chat"
        User: "I want to keep fit" → Intent: "chat"
        User: "I have an electronic voucher." → Intent: "chat"

        ⚠️ **Only return "voucher_query" or "chat", do NOT add any extra words.** ⚠️

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
            You are a voucher assistant who helps users inquire about available vouchers, how to use them, and where they can be redeemed.

            Objective:
            1. Provide accurate and concise answers to user queries regarding vouchers.
            2. Dynamically adjust responses to ensure clarity and avoid misleading information.
            3. Guide users to provide complete and relevant details for precise assistance.
            4. Maintain a friendly, engaging, and professional conversation tone.

            Voucher Categories:
            - Healthy Voucher
            - Environmental Voucher
            - Educational Voucher

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
            If the user enters an incomplete voucher name (e.g., "Healthy" instead of "Healthy Voucher"), prompt them to enter the full name:
            "To proceed, please enter the full voucher name (e.g., 'Healthy Voucher', 'Educational Voucher')."

            Providing Additional Information:
            If the user asks where a voucher can be used:
            "The Healthy Voucher can be used for purchasing health-related products. Would you like to see eligible stores?"
            If the user asks how to get a voucher:
            "You can obtain vouchers through our online platform or at participating stores. Would you like me to guide you through the process?"

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
            """


            # **加入历史对话**
            messages = [{"role": "system", "content": chat_prompt}] + chat_history

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=50
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