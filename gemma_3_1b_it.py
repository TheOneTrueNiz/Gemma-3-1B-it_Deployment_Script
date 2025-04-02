# Gemma-3-1B-it_Deployment_Script
# Simple Gemma-3-1B-it Deployment Script

# -*- Project: Gemma-3-1b-it Stateless Chatbot / NizBot Team -*-
# -*- Version: V1.1 - Text-Only Fix -*- # <-- Updated Version/Purpose

import gradio as gr
import torch
# Removed AutoProcessor, kept AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Removed PIL Image as it's text-only now
# from PIL import Image
import time
import os
import logging
import re
# Removed Optional/Image related types where not needed
from typing import List, Dict, Tuple, Any

# --- Type Alias ---
GradioHistory = List[Dict[str, Any]] # Keep this for Gradio Chatbot structure

# --- Configuration ---
# Model ID
MAIN_MODEL_NAME = "google/gemma-3-1b-it"

# Generation Settings
MAIN_MODEL_MAX_NEW_TOKENS = 2048
GENERATION_TEMP = 0.7
GENERATION_TOP_P = 0.9

# Context Management
MAX_HISTORY_TURNS = 5

# Optimization & Placement Settings
ATTN_IMPLEMENTATION = "flash_attention_2" # Keep or change based on your hardware/setup
QUANTIZATION_BITS = 8 # Keep or change (e.g., 4 or None for bf16/fp16)
# TARGET_EMBEDDING_DEVICE = "cuda:0" # Not explicitly used anymore, can be removed or ignored

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print(f"--- Script Starting (single_modelV1_gemma1B_{QUANTIZATION_BITS}bit_text_only.py) ---") # <-- Updated Filename

# --- Global Variables ---
main_model = None
# Removed: processor = None
tokenizer = None

# --- Model Loading Function (Simplified) ---
def load_model():
    """Loads the main Causal LM model and tokenizer."""
    # Removed processor from global scope modification
    global main_model, tokenizer
    logger.info("--- Starting Model Loading ---")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU acceleration and quantization.")

    logger.info(f"CUDA GPUs detected: {torch.cuda.device_count()}")
    main_model_dtype = torch.bfloat16
    load_8bit_flag = QUANTIZATION_BITS == 8
    load_4bit_flag = QUANTIZATION_BITS == 4

    quantization_config = None
    if load_8bit_flag:
        logger.info("8-bit Quantization requested.")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_4bit_flag:
        logger.info("4-bit Quantization requested.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif QUANTIZATION_BITS is not None:
        logger.warning(f"Unsupported QUANTIZATION_BITS value ({QUANTIZATION_BITS}). Loading in BF16.")
    else:
        logger.info("No quantization requested. Loading in BF16.")

    try:
        # Only load tokenizer now
        logger.info(f"Loading tokenizer for: {MAIN_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_NAME, trust_remote_code=True)
        # Removed: processor loading line

        logger.info(f"Loading main model: {MAIN_MODEL_NAME} (Target dtype: {main_model_dtype}, Quantized: {QUANTIZATION_BITS}-bit)")
        logger.info(f"Using device_map='auto' for automatic multi-GPU placement. Attn: {ATTN_IMPLEMENTATION}")

        t_start_main = time.time()
        main_model = AutoModelForCausalLM.from_pretrained(
            MAIN_MODEL_NAME,
            torch_dtype=main_model_dtype if not (load_4bit_flag or load_8bit_flag) else None,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
            trust_remote_code=True
        )
        t_end_main = time.time()
        logger.info(f"Main model loaded in {t_end_main - t_start_main:.2f} seconds.")

        # Set pad token if missing (Gemma often doesn't have one set by default)
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Also update model config if possible/needed, although generate uses tokenizer's pad_token_id
            main_model.config.pad_token_id = tokenizer.eos_token_id


        try:
            logger.info(f"Main model effective device map: {main_model.hf_device_map}")
            logger.info(f"Device of first parameter: {next(main_model.parameters()).device}")
        except (AttributeError, StopIteration):
            logger.warning("Could not retrieve detailed device map or parameter device.")

        logger.info("Model and Tokenizer ready.")
        logger.info("--- Model Loading Complete ---")

    except ImportError as ie:
         logger.error(f"ImportError during model loading: {ie}. A required library might be missing.", exc_info=True)
         if "flash_attn" in str(ie).lower() and ATTN_IMPLEMENTATION == "flash_attention_2":
             logger.error("FlashAttention-2 implementation specified but library not found/compatible. Try `pip install flash-attn --no-build-isolation` or change ATTN_IMPLEMENTATION.")
         elif "bitsandbytes" in str(ie).lower() and (load_8bit_flag or load_4bit_flag):
             logger.error("BitsAndBytes library not found/compatible. Install it (`pip install bitsandbytes`) for quantization.")
         raise
    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}", exc_info=True)
        raise


# --- Main Prediction Function (Text-Only) ---
def predict(
    user_input_text: str, # Removed Optional, assume text input is primary
    history: GradioHistory
# Removed user_input_image parameter
) -> Tuple[GradioHistory, str]: # Return type changed, no image output
    """Generates the chatbot response based on user text input and limited history."""
    # Removed processor from global check
    global main_model, tokenizer
    logger.info("--- Entering predict function ---")
    if history is None: history = []

    # Check essential components (model and tokenizer)
    # Removed processor from check
    if not all([main_model, tokenizer]):
        logger.error("A required component (model, tokenizer) is not loaded/initialized.")
        error_message = {"role": "assistant", "content": "Error: Critical component missing. Please restart."}
        # Add user input to history before error if present
        if user_input_text:
             history.append({"role": "user", "content": user_input_text})
        history.append(error_message)
        # Return history with error, clear text input
        return history, ""

    user_input_text_orig = user_input_text or ""
    # Removed pil_image and has_image logic
    has_text = bool(user_input_text_orig and user_input_text_orig.strip())
    logger.info(f"Processing user input. Text present: {has_text}")

    # Now only require text input
    if not has_text:
        logger.warning("Empty text input received.")
        # Return current history, clear input field
        return history, ""


    # Construct history entry for Gradio display (text only)
    history.append({"role": "user", "content": user_input_text_orig})

    # Construct API message content (text only)
    api_user_message_content = [{"type": "text", "text": user_input_text_orig}]

    # --- Prepare System Prompt and History for LLM (Simplified) ---
    system_prompt_text = (
        f"You are {MAIN_MODEL_NAME}, a helpful AI assistant created by Google, running locally. "       
        "Answer the user's current query based on the available context. Be helpful."
    )
    logger.debug(f"System Prompt:\n{system_prompt_text}")

    # Construct message history for the API call (limit turns)
    api_history_messages = []
    history_turns_to_consider = history[:-1]
    turns_added = 0
    for i in range(len(history_turns_to_consider) - 1, -1, -1):
        turn = history_turns_to_consider[i]
        role = turn.get("role")
        content_str = str(turn.get("content", "")).strip() # Ensure it's string and stripped

        # Removed image stripping logic as images aren't included
        if not content_str: # Skip empty turns
            continue

        if role == "user":
            api_history_messages.insert(0, {'role': 'user', 'content': [{'type': 'text', 'text': content_str}]})
            turns_added += 1
        elif role == "assistant":
            # API expects model role content as string for text-only
            api_history_messages.insert(0, {'role': 'model', 'content': content_str})

        if turns_added >= MAX_HISTORY_TURNS:
            logger.info(f"Reached MAX_HISTORY_TURNS ({MAX_HISTORY_TURNS}). Truncating history.")
            break

    # Prepare the final list of messages for the API
    messages_for_api = [
        {'role': 'user', 'content': [{'type': 'text', 'text': system_prompt_text}]},
        {'role': 'model', 'content': "Okay, I've reviewed the context. What can I help you with?"}
    ] + api_history_messages + [
        {'role': 'user', 'content': api_user_message_content}
    ]
    logger.debug(f"Messages for API (first/last few):\nSystem: {messages_for_api[0]}\n...\nUser: {messages_for_api[-1]}")


    # --- LLM Generation ---
    response_text = "Error: Failed to generate response."
    try:
        logger.info("Applying chat template and tokenizing inputs...")
        t1 = time.time()

        # Use tokenizer to apply chat template
        prompt_for_model = tokenizer.apply_chat_template(messages_for_api, tokenize=False, add_generation_prompt=True)

        # Use tokenizer to prepare model inputs (text only)
        # Removed images=pil_image
        inputs = tokenizer(prompt_for_model, return_tensors="pt").to(main_model.device)

        logger.info(f"Tokenization/Processing took {time.time()-t1:.2f}s. Generating response...")

        generation_start_time = time.time()
        with torch.no_grad():
            outputs = main_model.generate(
                **inputs,
                max_new_tokens=MAIN_MODEL_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GENERATION_TEMP,
                top_p=GENERATION_TOP_P,
                # Use tokenizer's pad token id
                pad_token_id=tokenizer.pad_token_id
            )
        generation_end_time = time.time()
        gen_duration = generation_end_time - generation_start_time
        logger.info(f"Main model generation finished in {gen_duration:.2f} seconds.")

        # Decode only the newly generated tokens
        input_token_len = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_token_len:]
        # Use tokenizer to decode
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        num_generated_tokens = len(response_tokens)
        tokens_per_sec = num_generated_tokens / gen_duration if gen_duration > 0 else 0
        logger.info(f"Generated {num_generated_tokens} tokens ({tokens_per_sec:.2f} tokens/sec).")
        logger.info(f"Generated response (decoded, first 150 chars): '{response_text[:150]}...'")

        # Append final assistant response to history for Gradio output
        history.append({"role": "assistant", "content": response_text})

    except Exception as e:
        logger.error(f"Error during main model prediction or response handling: {e}", exc_info=True)
        error_content = f"Sorry, I encountered an error generating the response: {type(e).__name__}. Please check logs."
        if history and history[-1]['role'] == 'user':
            history.append({"role": "assistant", "content": error_content})
        else:
            logger.error("Unexpected history state before adding error message.")
            history.append({"role": "assistant", "content": error_content})

    logger.info("--- Exiting predict function ---")
    # Return history and clear text input field
    return history, ""


# --- Gradio Interface (Text-Only) ---
def create_gradio_interface():
    """Creates and configures the Gradio web interface."""
    logger.info("--- Creating Gradio Interface ---")
    with gr.Blocks(theme=gr.themes.Soft(), title="Gemma-3 1B Local Chatbot (Niz/Gemini Project - Text-Only)") as demo:
        gr.Markdown(f"# 🤖 Gemma-3 1B Local Chatbot (Stateless V1.1 Text-Only - {QUANTIZATION_BITS}-bit)")
        gr.Markdown("*(Your Lightweight Collaborative AI Bud!)*")
        gr.Markdown(f"**Model:** `{MAIN_MODEL_NAME}` (Quantized: {QUANTIZATION_BITS}-bit, Attn: {ATTN_IMPLEMENTATION}, Device: auto)")

        # Simplified Status Check (model and tokenizer only)
        # Removed processor check
        status_msg = "All Systems Go!" if all([main_model, tokenizer]) else "ERROR: Critical component(s) failed! Check logs."
        status_style = "color:green; font-weight:bold;" if all([main_model, tokenizer]) else "color:red; font-weight:bold;"
        gr.Markdown(f"<p style='{status_style}'>**Status:** {status_msg}</p>")

        # Updated chatbot component call
        chatbot = gr.Chatbot(
            label="Conversation",
            height=650,
            show_copy_button=True,
            type="messages" # Use recommended type
            # Removed bubble_full_width
        )

        with gr.Row():
            # Removed image_input component
            # text_input takes full width now potentially, or adjust scale
            text_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message...",
                scale=4, # Adjust scale as needed
                interactive=True,
                autofocus=True
            )

        with gr.Row():
            submit_button = gr.Button("➡️ Send", variant="primary", scale=3)
             # Removed image_input from clear list
            clear_button = gr.ClearButton([text_input, chatbot], value="🗑️ Clear All", scale=1)

        # Define inputs and outputs for the predict function (text-only)
        # Removed image_input
        submit_inputs = [text_input, chatbot]
         # Removed image_input from outputs
        submit_outputs = [chatbot, text_input]

        # Link buttons/actions to the predict function
        submit_button.click(fn=predict, inputs=submit_inputs, outputs=submit_outputs, api_name="predict")
        text_input.submit(fn=predict, inputs=submit_inputs, outputs=submit_outputs, api_name="predict_submit")

    logger.info("--- Gradio Interface Created ---")
    return demo

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Main Execution Starting ---")

    # Load the model
    try:
        load_model()
        logger.info("Model loading successful.")
    except Exception as e:
        logger.error(f"Fatal Error: Model loading failed ({type(e).__name__}). Exiting.", exc_info=True)
        try:
             app = create_gradio_interface()
             logger.warning("--- Launching Gradio with Model Loading Error ---")
             app.queue().launch(server_name="0.0.0.0")
        except Exception as ge:
             logger.error(f"Fatal Error: Failed to launch Gradio interface even after model error: {ge}", exc_info=True)
        exit(1)

    # Create and launch the Gradio interface
    try:
        app = create_gradio_interface()
        logger.info("--- Launching Gradio Application ---")
        app.queue().launch(server_name="0.0.0.0")
    except Exception as e:
        logger.error(f"Fatal Error: Failed to create or launch Gradio interface: {e}", exc_info=True)
        exit(1)

    logger.info("--- Gradio Application Closed ---")
    logger.info("--- Main Execution Finished ---")
