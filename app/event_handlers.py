# ============================================================================
# Copyright 2023 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
# ============================================================================

"""This module defines handlers(methods) to handle event's raised by UI components."""

import configparser
from google.cloud import speech
import gradio as gr
from models.session_info import SessionInfo
from middleware import app, progress, registrasi, cek_akun
from langchain_core.messages import HumanMessage, ToolMessage
from pydub import AudioSegment


class EventHandlers:
  """A class to define event handlers for chat bot UI components.

  Attributes:
        config: A configparser having gradio configurations loaded from
          config.ini file.
  """

  config: configparser.ConfigParser()

  def __init__(self, config: configparser.ConfigParser):
    self.config = config

  def transcribe_file(self, speech_file: str) -> str:
    """Transcribe the audio file and return the converted text.

    Args:
        speech_file (str): Path to speech file.

    Returns:
        text (str): Transcribed text from the input speech.
    """
    text = ""

    # Initialize the speech client
    client = speech.SpeechClient()

    # Load the audio file using pydub
    audio = AudioSegment.from_wav(speech_file)

    # Convert stereo to mono if the audio is stereo (2 channels)
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Export the mono audio to a temporary file (in memory)
    with open("temp_mono_audio.wav", "wb") as temp_file:
        audio.export(temp_file, format="wav")

    # Read the audio content from the temporary file
    with open("temp_mono_audio.wav", "rb") as audio_file:
        content = audio_file.read()

    # Prepare the audio and config for the Google Speech-to-Text API
    audio = speech.RecognitionAudio(content=content)

    # Set the correct sample rate based on your file's sample rate (48000 Hz)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,  # Use the correct sample rate from your WAV file
        language_code="en-US",
    )

    # Make the API call to recognize the speech
    response = client.recognize(config=config, audio=audio)

    # Iterate through the results and get the transcriptions
    for result in response.results:
        # Get the most likely transcript from the alternatives
        text = result.alternatives[0].transcript

    return text

  def add_user_input(self, history, text):
    """Adds user input to chat history.

    Args:
            history (dict): A dictionary that stores user's chat.
            text (str): String input to chat bot.

    Returns:
            history (dict): A dictionary that stores user's chat.
    """
    if bool(text):
      history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

  def clear_history(self, history, session):
    """Clear chat history.

    Args:
            history (dict): A dictionary that stores user's chat.
            session: current session object.

    Returns:
            history (dict): A dictionary that stores user's chat.
            session: current session object.
            source_location(None): To clear source location textbox.
    """
    history = [(None, self.config["initial-message"])]
    session = []
    session.append(SessionInfo())
    return history, session, None

  def bot_response(self, history, session):
    """Returns session, source location and chat history with the updated Bot response.

    Args:
            history (dict): A dictionary that stores user's chat.
            session: current session object.

    Returns:
            history (dict): A dictionary that stores user's chat.
            session: current session object.
            source_location: string representing manual/spec location. Usually a
            gcs link.
    """
    if not session:
      session.append(SessionInfo())

    session_info: SessionInfo = session[0]
    response = ""
    source_location = ""
    if history[-1][0] is None:
      return history, session
    else:
      response = app.invoke(
        {"messages": [HumanMessage(content=history[-1][0])]},
        config={"configurable": {"thread_id": "1"}},
      )

      last_message = response['messages'][-1]
      last_message_content = last_message.content

      for tool_call in last_message.tool_calls:
        selected_tool = {"progress": progress, "registrasi": registrasi, "cek_akun": cek_akun}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        last_message_content = tool_output

      model_response = last_message_content
      if model_response is None:
        response = self.config["error-response"]
      else:
        response = model_response
        source_location = ''

    if not bool(response):
      response = self.config["error-response"]

    history[-1][1] = response
    return (history, session, source_location)
