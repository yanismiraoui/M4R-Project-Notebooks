# Import all the necessary modules
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sonnia.processing import Processing
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Concatenate
from sklearn.decomposition import PCA

def train_model_basic(df, X_train, params={}):
    if isinstance(X_train, list):
        X_train = X_train[0]
    # define encoder
    visible = Input(shape=(params["n_inputs"],))
    e = Dense(params["n_inputs"]*2)(visible)
    e = BatchNormalization()(e)
    e = ReLU()(e)
    # define bottleneck
    n_bottleneck = 2
    bottleneck = Dense(n_bottleneck)(e)
    # define decoder
    d = Dense(params["n_inputs"]*2)(bottleneck)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    # output layer
    output = Dense(params["n_inputs"], activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=1)

    # Return encoder part of the model
    encoder = Model(inputs=visible, outputs=bottleneck)

    #Plot the history loss of the model
    plt.plot(history.history['loss'], label='train')
    plt.title('Model loss')
    plt.legend()
    plt.show()

    return model, encoder

def train_model_complex(df, X_train, params={}):
    # define encoder
    cdr3_input = Input(shape=(params["n_inputs"],), name='cdr3_input')
    v_gene_input = Input(shape=(params["v_inputs"],), name='v_gene_input')
    j_gene_input = Input(shape=(params["j_inputs"],), name='j_gene_input')
    cdr3_embedding = Dense(params["n_inputs"]*2)(cdr3_input)
    cdr3_embedding = BatchNormalization()(cdr3_embedding)
    cdr3_embedding = ReLU()(cdr3_embedding)

    v_gene_embedding = Dense(params['v_gene_embedding_dim'], name='v_gene_embedding')(v_gene_input)
    j_gene_embedding = Dense(params['j_gene_embedding_dim'], name='j_gene_embedding')(j_gene_input)
    merged_embedding = Concatenate(axis=1,name='merged_embedding')([cdr3_embedding, v_gene_embedding, j_gene_embedding])

    encoder_dense_1 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_1')(merged_embedding)
    encoder_dense_2 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_2')(encoder_dense_1)

    # Latent layers:
    bottleneck = Dense(params['latent_dim'], name='bottleneck')(encoder_dense_2)

    # define decoder
    d = Dense(params["n_inputs"]*2)(bottleneck)
    d = BatchNormalization()(d)
    d = ReLU()(d)

    # output layer
    output = Dense(params["n_inputs"], activation='linear')(d)
    # define autoencoder model
    model = Model([cdr3_input, v_gene_input, j_gene_input], [output])
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=params["epochs"], batch_size=64, verbose=1)

    # Return encoder part of the model
    encoder = Model(inputs=[cdr3_input, v_gene_input, j_gene_input], outputs=bottleneck)
    decoder = Model(inputs=bottleneck, outputs=output)

    #Plot the history loss of the model
    plt.plot(history.history['loss'][10:], label='complex model', color='blue')
    plt.title('Model loss')
    plt.legend()
    plt.show()


    return model, encoder, decoder



def model_results(df, X_test, encoder, label="label", mode_complex=True):
    if label == "end_seq_label":
        df[label] = df["CDR3_al"].apply(lambda x: x[15:])
    elif label ==  "begin_seq_label":
        df[label] = df["CDR3_al"].apply(lambda x: x[:5])
    elif label == "j_gene":
        df[label] = df["j_gene"].apply(lambda x: x.split("-")[0])

    
    labels = []
    labels_encoder = LabelEncoder()
    labels_encoder = labels_encoder.fit(df[label].unique())
    for k in tqdm(df.index):
        labels.append(labels_encoder.transform([df.loc[k,label]]))
    labels = [int(y) for y in labels]
    df[label] = labels
    rgb_values = sns.color_palette("Spectral", df[label].nunique())
    df[str(label+"_color")] = df[label].apply(lambda x: rgb_values[x])
    N = 300
    if mode_complex:
        X_test = X_test.copy()
        for i in range(len(X_test)):
            X_test[i] = X_test[i][:N]
        X_test_encode = encoder.predict(X_test[:N])
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X_test_encode)
        plot_X_test = principalComponents
    else:
        if isinstance(X_test, list):
            X_test = X_test[0]
        X_test_encode = encoder.predict(X_test[:N])
        plot_X_test = X_test_encode.copy()

    print(len(X_test_encode))
    print(X_test_encode.shape)
    plt.scatter(plot_X_test[:,0], plot_X_test[:,1], color=df[str(label+"_color")][:N], alpha=0.5)
    plt.title(label)
    plt.show()


    k = 8
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_test_encode[:N])
    labels = pd.Series(kmeans.labels_)
    rgb_values = sns.color_palette("Spectral", k)
    col_kmeans = labels.apply(lambda x: rgb_values[x])
    sample = df[:N]
    sample[str(label+"_kmeans_label")] = kmeans.labels_
    label_dict = {}
    for cluster in range(k):
        label_ind = sample[sample[str(label+"_kmeans_label")] == cluster][label].value_counts().index[0]
        label_dict[cluster] = label_ind

    print("Accuracy: ", sum([label_dict[x] == y for x,y in zip(sample[str(label+"_kmeans_label")], sample[label])])/len(sample))
    plt.scatter(plot_X_test[:,0], plot_X_test[:,1], color=col_kmeans, alpha=0.5)
    centroids = kmeans.cluster_centers_
    if not mode_complex:
        for cluster in range(k):
            plt.text(centroids[cluster,0], centroids[cluster,1], labels_encoder.inverse_transform([label_dict[cluster]]), fontsize=10)
    plt.title("Kmeans clustering for "+label)
    plt.show()
