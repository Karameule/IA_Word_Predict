import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Définir une classe de jeu de données personnalisée
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                words = line.split()
                input_indices = [word_to_index[word] for word in words[:2]]
                output_index = word_to_index[words[2]]
                self.data.append((torch.tensor(input_indices), torch.tensor(output_index)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Charger le fichier et créer un vocabulaire
file_path = 'dataset-test-1-AI.txt'
word_to_index = {}
index = 0
with open(file_path, 'r') as file:
    for line in file:
        words = line.split()
        for word in words:
            if word not in word_to_index:
                word_to_index[word] = index
                index += 1

# Définir la classe du modèle
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.view(x.size(0), -1)  # Aplatir les embeddings
        x = F.relu(self.fc1(embedded))
        x = self.fc2(x)
        return x

# Initialiser le jeu de données et le dataloader
dataset = CustomDataset(file_path)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Définir les hyperparamètres
vocab_size = len(word_to_index)
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size
num_epochs = 45
learning_rate = 0.001

# Instancier le modèle
model = MyModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Définir l'optimiseur et la fonction de perte
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Boucle d'entraînement
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
        
        # Calculer le nombre de prédictions correctes dans le batch
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}')
    
    # Calculer le loss moyen et la précision pour l'epoch
    epoch_loss /= len(dataset)
    accuracy = correct_predictions / total_predictions
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {accuracy*100}%')



########################### Test avec mots choisis ###########################

# Supposons que 'a' et 'a' soient les deux premiers mots que vous voulez utiliser
premier_mot = 'our'
deuxieme_mot = 'is'

# Construire un dictionnaire qui mappe les index des mots vers les mots correspondants
index_to_word = {index: word for word, index in word_to_index.items()}

# Trouver les indices correspondant aux deux premiers mots
index_premier_mot = word_to_index[premier_mot]
index_deuxieme_mot = word_to_index[deuxieme_mot]

# Imprimer les deux premiers mots
print(f"Les deux premiers mots sont : {premier_mot} {deuxieme_mot}")

# Utiliser les indices des deux premiers mots pour prédire le troisième mot
input_indices = [index_premier_mot, index_deuxieme_mot]
input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Ajouter une dimension de batch

# Utiliser le modèle pour prédire le troisième mot
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    _, predicted_index = torch.max(probabilities, dim=1)

# Convertir l'index prédit en mot
predicted_word = index_to_word[predicted_index.item()]

print(f"Le troisième mot prédit est : {predicted_word}")

###########################       PCA       #####################################

# Conversion des embeddings en numpy array
embeddings = []
for idx in range(vocab_size):
    embedding_tensor = model.embedding(torch.tensor(idx)).detach().numpy()
    embeddings.append(embedding_tensor)
embeddings = np.array(embeddings)

# Réduction de dimension avec PCA
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

# Affichage des embeddings réduits avec PCA
plt.figure(figsize=(10, 8))
plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], alpha=0.5)
for word, index in word_to_index.items():
    plt.annotate(word, (pca_embeddings[index, 0], pca_embeddings[index, 1]), fontsize=8)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Word Embeddings Visualization with PCA')
plt.grid(True)
plt.show()