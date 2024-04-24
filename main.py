import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tkinter import Tk, Label, Entry, Button, Menu

def load_data(filename):
  """Loads data from a CSV file."""
  data = pd.read_csv(filename)
  questions = data['question'].tolist()
  answers = data['answer'].tolist()
  return questions, answers

def preprocess_data(questions, answers):
  """Preprocesses text data for the LSTM model."""
  tokenizer = Tokenizer(num_words=10000)
  tokenizer.fit_on_texts(questions + answers)

  question_sequences = tokenizer.texts_to_sequences(questions)
  answer_sequences = tokenizer.texts_to_sequences(answers)

  max_length = max([len(seq) for seq in question_sequences])
  question_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')
  answer_padded = pad_sequences(answer_sequences, maxlen=max_length, padding='post')

  return tokenizer, question_padded, answer_padded

def generate_answer(model, tokenizer, question_padded):
  """Generates an answer using the trained LSTM model."""
  predicted_answer = model.predict(question_padded)[0]
  predicted_answer_index = np.argmax(predicted_answer)
  predicted_answer_text = tokenizer.index_word[predicted_answer_index]
  return predicted_answer_text

def create_gui(model_filename=None, tokenizer_filename=None):
  """Creates the graphical user interface (GUI)."""
  global model, tokenizer, question_padded

  root = Tk()
  root.title('Chatbot')

  def update_dataset(filename):
    global model, tokenizer, question_padded
    questions, answers = load_data(filename)
    tokenizer, question_padded, answer_padded = preprocess_data(questions, answers)
    # If a model file exists, load it
    if model_filename is not None:
      model = tf.keras.models.load_model(model_filename)
    else:
      model = tf.keras.Sequential([
          Embedding(10000, 128, input_length=256),
          LSTM(128, return_sequences=True),
          LSTM(128),
          Dense(10000, activation='softmax')
      ])
      model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      model.fit(question_padded, answer_padded, epochs=10, batch_size=32)
      model.save('chatbot_model.h5')  # Replace with your desired filename

  menubar = Menu(root)
  dataset_menu = Menu(menubar, tearoff=0)
  dataset_menu.add_command(label="Dataset 1 (data.csv)", command=lambda: update_dataset("data.csv"))

  dataset_menu.add_separator()
  dataset_menu.add_command(label="Exit", command=root.quit)
  menubar.add_cascade(label="Dataset", menu=dataset_menu)
  root.config(menu=menubar)

  user_input_label = Label(root, text='Введите вопрос:')
  user_input_label.grid(row=0, column=0, sticky='w')

  user_input_entry = Entry(root)
  user_input_entry.grid(row=0, column=1, sticky='ew')

  answer_label = Label(root, text='Ответ:')
  answer_label.grid(row=1, column=0, sticky='w')

  answer_entry = Entry(root, state='disabled')
  answer_entry.grid(row=1, column=1, sticky='ew')

  send_button = Button(root, text='Отправить',)
