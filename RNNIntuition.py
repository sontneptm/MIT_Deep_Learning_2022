
"""this is pseudocode"""

my_rnn = RNN()

hidden_state = [0,0,0,0]
sentence = ["I", "love", "recurrent", "neural"] # it's not a set of inputs, but an input.

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction