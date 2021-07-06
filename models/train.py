# train/test pickle로 따로 저장
path = './data/final_data/0521_ked_train_only(full_leastleast_update_token).pkl'
test_path = './data/final_data/0512_ked_test_only(update_token).pkl'
batch_size =  128
train_sentences, train_labels = read_corpus(path)
ked_test_sentences, ked_test_labels = read_test_corpus(test_path)

### train
train_padded_sentences = preprocess(train_sentences)
train_dataset = Dataset(train_padded_sentences, train_labels, batch_size)

## test
test_padded_sentences = preprocess(test_sentences)
test_batch_size = batch_size   # train과 동일하게 설정
test_dataset = Dataset(test_padded_sentences, test_labels, test_batch_size)

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(None, EMBEDDING_DIM)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    model.add(tf.keras.layers.Dense(100, kernel_initializer= 'he_normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(train_dataset[0][1].shape[1], activation='softmax'))    ## 17개의 대분류 분류
    return model

model = build_model()
batch_size =  128
epoch = 7
adam = Adam(lr=3e-3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(train_dataset, epochs=epoch)
model.save('./model/FINAL_MODELS/direct_final_small_0522_wiki_emb2.h5')