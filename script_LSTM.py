import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Función para convertir una secuencia de ADN a una matriz 350x4
def convertir_secuencia(secuencia):
    matriz = np.zeros((len(secuencia), 4), dtype=int)
    for i, letra in enumerate(secuencia):
        matriz[i] = letra_a_vector[letra]
    return matriz


class CustomDataset(Dataset):
    def __init__(self, carpeta_datos, carpetas):
        self.X, self.y = self._cargar_datos(carpeta_datos, carpetas)

    def _cargar_datos(self, carpeta_datos, carpetas):
        X = []
        y = []

        for carpeta in carpetas:
            carpeta_actual = os.path.join(carpeta_datos, carpeta)
            archivos = os.listdir(carpeta_actual)

            for archivo in archivos:
                path = os.path.join(carpeta_actual, archivo)
                with open(path, 'r') as archivo:
                    for linea in archivo:
                        elementos = linea.strip().split()
                        secuencia_adn = elementos[0]
                        valor_arnt =  float(elementos[1])
                        matriz_secuencia = convertir_secuencia(secuencia_adn)
                        X.append(matriz_secuencia)
                        y.append(valor_arnt)

        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

        
            


class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Capa LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Capa de salida con función de activación Sigmoid
        #self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Propagación a través de la capa LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Capa de salida
        #out = self.dropout(out[:, -1, :])
        out = self.fc(out[:, -1, :])  # Tomar el último paso de tiempo como entrada
        #out = self.fc(out) 
        predictions = self.sigmoid(out)  # Sigmoid en lugar de Softmax

        return predictions
    
# Función de entrenamiento

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    epoch_mse = 0.0
    epoch_mae = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # Utilizar la función de pérdida de PyTorch para ambas métricas
        mse_loss = criterion(outputs, labels.unsqueeze(1))
        mae_loss = nn.L1Loss()(outputs, labels.unsqueeze(1))
        mse_loss.backward()
        optimizer.step()

        # Acumular pérdidas ponderadas por el tamaño del lote
        batch_size = inputs.size(0)
        total_samples += batch_size
        epoch_mse += mse_loss.item() * batch_size
        epoch_mae += mae_loss.item() * batch_size

    average_mse = epoch_mse / total_samples
    average_mae = epoch_mae / total_samples

    print(f"Epoch MSE: {average_mse:.4f}, MAE: {average_mae:.4f}")

    # Calcular la referencia de MAE como la media de las etiquetas
    reference_mae = np.mean(np.array(labels)) 

    return average_mse, average_mae, reference_mae

def evaluate_epoch(model, dataloader, criterion, reference):
    model.eval()
    epoch_mse = 0.0
    epoch_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            y_pred_tensor = model(torch.tensor(np.array(inputs), dtype=torch.float32))

            mse_loss = criterion(y_pred_tensor, torch.tensor(labels, dtype=torch.float32).unsqueeze(1))
            mae_loss = nn.L1Loss()(y_pred_tensor, torch.tensor(labels, dtype=torch.float32).unsqueeze(1))

            batch_size = inputs.size(0)
            total_samples += batch_size

            epoch_mse += mse_loss.item() * batch_size
            epoch_mae += mae_loss.item() * batch_size

            # Append targets and predictions for later plotting


    average_mse = epoch_mse / total_samples
    average_mae = epoch_mae / total_samples

    print(f"Eval MSE: {average_mse:.7f}, MAE: {average_mae:.4f}")

    #Mae entre referencia y etiqueta real
    constant_prediction_mae = np.mean(np.abs(np.array(reference) - np.array(dataloader.dataset.y)))
    constant_prediction_mse = np.mean((np.array(reference) - np.array(dataloader.dataset.y)) ** 2)



    return average_mse, average_mae, constant_prediction_mae, constant_prediction_mse


# Función principal de entrenamiento
def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs=10, plot_dir='/opt/Experimentos/SV/grafica/', model_dir='/opt/Experimentos/SV/modelo/'):


    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #variables train
    train_mse_losses = []
    train_mae_losses = []
    train_reference_mae=[]
    #variables validation
    val_mse_losses = []
    val_mae_losses = []
    reference_mae=[]
    reference_mse=[]


    for epoch in range(num_epochs):

        #Train Model
        train_mse, train_mae, train_reference = train_epoch(model, dataloader_train, criterion, optimizer)
        train_mse_losses.append(train_mse)
        train_mae_losses.append(train_mae)
        train_reference_mae.append(train_reference)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train MSE: {train_mse:.4f}, MAE: {train_mae}")

        #Validation Model
        val_mse, val_mae, val_reference, mse_ref= evaluate_epoch(model, dataloader_val, criterion, train_reference)
        val_mse_losses.append(val_mse)
        val_mae_losses.append(val_mae)
        reference_mae.append(val_reference)
        reference_mse.append(mse_ref)



        #model_name = f'lstm_model_epoch_{epoch + 1}.pth'
        #model_path = os.path.join(model_dir, model_name)
        #torch.save(model.state_dict(), model_path)
        #print(f'Modelo guardado en: {model_path}')


        print(f"Epoch {epoch + 1}/{num_epochs} - Validation MSE: {val_mse:.4f}, MAE: {val_mae}")


        print("---------------------------")



    # Plot and save MSE for each epoch
    reference_m=np.mean(np.array(reference_mse))
    print(reference_mse)

    plt.plot(range(1, epoch + 2), train_mse_losses, marker='o', linestyle='-', color='b', label='MSE Train')
    plt.plot(range(1, epoch + 2), val_mse_losses, marker='o', linestyle='-', color='r', label='MSE Validation')
    plt.axhline(y=reference_m, color='black', linestyle='--', label='MSE Reference (Constant Prediction)')
    plt.title('Mean Squared Error (MSE) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'mse_epoch_{epoch + 1}.png'))
    plt.close()

    reference=np.mean(np.array(reference_mae))
    print(reference)
    # Plot and save MAE for each epoch
    plt.plot(range(1, epoch + 2), train_mae_losses, marker='o', linestyle='-', color='b', label='MAE Train')
    plt.plot(range(1, epoch + 2), val_mae_losses, marker='o', linestyle='-', color='r', label='MAE Validation')
    plt.axhline(y=reference, color='black', linestyle='--', label='MAE Reference (Constant Prediction)')
    plt.title('Mean Absolute Error (MAE) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'mae_epoch_{epoch + 1}.png'))
    plt.close()


    print("Finished training")



if __name__ == "__main__":
    BATCH_SIZE = 50
    EPOCHS = 10
    LEARNING_RATE = 0.001

    directorio_datos = '/opt/Experimentos/SV/Datos/'

    letra_a_vector = {'A': [1, 0, 0, 0],'N': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

    # Obtener la lista de carpetas en el directorio
    carpetas = [nombre for nombre in os.listdir(directorio_datos) if os.path.isdir(os.path.join(directorio_datos, nombre))]

    # Dividir en entrenamiento, validación y test
    carpetas_train, carpetas_test = train_test_split(carpetas, test_size=0.3, random_state=42)
    carpetas_train, carpetas_val = train_test_split(carpetas_train, test_size=0.33, random_state=42)

    print('Cargando Dataset Train...\n')
    dataset_train = CustomDataset(directorio_datos, carpetas_train)
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    print('Cargando Dataset Validation...\n')
    dataset_val = CustomDataset(directorio_datos, carpetas_val)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    print('Cargando Dataset Test...\n')
    #dataset_test = CustomDataset(directorio_datos, carpetas_test)
    #test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    
    input_size = 4
    hidden_size = 64
    num_layers = 3
    output_size = 1

    print('Creando Modelo LSTM...\n')
    model = LSTModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Entrenar el modelo
    print('Entrenamiento...\n')
    train_model(model, train_dataloader,val_dataloader, criterion, optimizer, num_epochs=EPOCHS)

    # Evaluar el modelo en el conjunto de prueba
    print('Evaluacion...\n')
    
    #evaluate(model, val_dataloader, criterion, num_epochs=EPOCHS)

    print('TERMINADO...\n')

