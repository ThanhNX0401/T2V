import openai
import nltk
import time
# Initialize the client
client = openai.OpenAI(api_key="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")


def paragraph_to_sentences(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def get_GPTprompt(user_prompt="A panda is eating bamboo",video_length=4):
    # Step 1: Create an Assistant
    assistant = client.beta.assistants.create(
        name="The Director",
        instructions=f"You are a director for a short clip who write {video_length} sentences paragraph to the customers regarding the scene they want you to make",
        model="gpt-3.5-turbo"
    )
    # Step 2: Create a Thread
    thread = client.beta.threads.create()

    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_prompt
    )
    print("running assistant")
    # Step 4: Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=f"""
            You are a director for a short clip who write {video_length} sentences paragraph to the customers regarding the scene they want you to make. Follow the instructions listed below:
            1. You have to identify the object, action of the given scene, visualize that action and write sentences describe it. The changing of motion in each sentences must be small.
            2. Later sentence must have the given subject from previous sentence.
            3. Be straightforward and do not use a narrative style, replace reflexive pronouns with their original vocabulary and eliminate the discourse cohesion while keeping the meaning the same.
            4. You must not use pronouns "It","They","He","His","Its" and replace it with the subject directly.
            5. Each sentence should be able to fully express all the visual information. Also, the linguistic structure of each sentence should be simple and similar.
            6. Don't describe the input in the first sentences, don't add any thing else to the answer except the {video_length} sentences paragraph.
            """
    )


    while True:
        # Wait for 5 seconds
        time.sleep(1)  

        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        # If run is completed, get messages
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            break
        
    prompt = messages.data[0].content[0].text.value
    nltk.download('punkt')  # Make sure to download the punkt tokenizer data

    # Convert each sentence into a string element and save them in a list
    sentences_list = paragraph_to_sentences(prompt)
    if len(sentences_list) > video_length:
      sentences_list = sentences_list[:video_length]
      
    elif len(sentences_list) < video_length:
        last_sentence = sentences_list[-1]  # Get the last element of sentences_list
        padding_length = video_length - len(sentences_list)
        sentences_list += [last_sentence] * padding_length
    # Print the result
    print(sentences_list)
    
    return sentences_list
