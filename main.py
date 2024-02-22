import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
st.set_option('deprecation.showPyplotGlobalUse', False)

class BreastCancerApp:
    def __init__(self):
        self.data = None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.clf = None
        self.params = dict()

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

        # datanın ilk 10 satırı
        st.subheader('Datanın ilk 10 satırı')
        st.write(self.data.head(10))

        st.subheader('Sütunlar')
        st.write(self.data.columns)

    def clean_and_preprocess_data(self):
        # Gereksiz colmunları temizleme işlemi
        self.data = self.data.drop(['Unnamed: 32', 'id','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se'], axis=1)

        # datanın son 10 satırı
        st.subheader('Datanın son 10 satırı')
        st.write(self.data.tail(10))

        # 'diagnosis' sütunundaki M değerini 1, B değerini 0 olarak değiştirme işlemi
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

        # 'diagnosis' sütunundaki güncellemeyi görmek için datanın tekrardan son 10 satırı
        st.subheader('Datanın son 10 satırı (diagnosis güncellemesi ile)')
        st.write(self.data.tail(10))

        # Diagnosis verisini Y olarak ve kalan veriyi ise X verisi olarak kullanma işlemi 
        Y = self.data['diagnosis']
        X = self.data.drop('diagnosis', axis=1)

        # Korelasyon matrisini çizdirme işlemi
        plt.figure(figsize=(12, 8))
        sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
        st.pyplot()
      
        # Malignant ve benign olacak şekilde datayı ayırma ve x:radiusmean y:texturemean olarak çizdirme işlemi
        malignant_data = self.data[self.data['diagnosis'] == 1]
        benign_data = self.data[self.data['diagnosis'] == 0]

        # Sütunlar arasındaki İlişki

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant_data, label='Kötü', color='red')
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign_data, label='İyi', color='blue')
        plt.title('Scatter Plot of Radius Mean vs Texture Mean')
        plt.xlabel('Radius Mean')
        plt.ylabel('Texture Mean')
        plt.legend()
        st.pyplot()

        sns.boxplot(x='diagnosis', y='radius_mean', data=self.data)
        plt.title(f'Box Plot of Radius Mean by Diagnosis')
        plt.xlabel('Diagnosis (M: 1, B: 0)')
        plt.ylabel('Radius Mean')
        st.pyplot()

        st.write('İyi huylu kanser hücrelerinin çekirdek boyutu, kötü huylu kanser hücrelerinin çekirdek boyutuna göre daha küçüktür.')

        sns.pairplot(self.data, hue='diagnosis', vars=['radius_mean', 'texture_mean','perimeter_mean', 'area_mean', 'smoothness_mean'])
        plt.suptitle('Pair Plot of Selected Features by Diagnosis', y=1.02)
        st.pyplot()

        plt.figure(figsize=(8, 4))
        g = sns.catplot(x='diagnosis', y='symmetry_mean', data=self.data, kind='bar', height=4)
        g.set_ylabels("Symmetry Mean")
        plt.title("Diagnosis vs Symmetry Mean")
        st.pyplot()
        st.write('Bu grafikten anlaşıldığı üzere kötü huylu kanser hücreleri iyi huylu kanser hücrelerine göre daha simetriktir.')

        plt.figure(figsize=(8, 4))
        g = sns.catplot(x='diagnosis', y='perimeter_mean', data=self.data, kind='bar', height=4)
        g.set_ylabels("Perimeter Mean")
        plt.title("Diagnosis vs Perimeter Mean")
        st.pyplot()
        st.write('Bu grafikten anlaşıldığı üzere kötü huylu kanser hücrelerinin çekirdek çevresi, iyi huylu kanser hücrelerinin çekirdek çevresine göre daha uzundur. ''perimeter_mean'' değeri ne kadar büyükse, çekirdek çevresi uzunluğu o kadar fazladır.')

        # Veriyi X_train, Y_train, X_test ve Y_test olarak yüzde 80-20 oranında ayırma işlemi
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        st.write('X_test:')
        st.write(self.X_test)
        st.write('Y_test:')
        st.write(self.Y_test)

        st.subheader('Model İmplementasyonu')

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def implement_model(self, classifier_name):
        self.classifier_name = classifier_name
        
        if self.classifier_name == 'KNN':
            param_grid = {'n_neighbors': list(range(1, 16))}
            self.clf = KNeighborsClassifier()
        elif self.classifier_name == 'SVM':
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            self.clf = SVC()
        elif self.classifier_name == 'Naive Bayes':
            param_grid = {'var_smoothing': [0.1, 0.5, 1.0]}
            self.clf = GaussianNB()
        else:
            st.write(f"Classifier {self.classifier_name} not supported.")
            return

        if self.classifier_name in ['KNN', 'SVM']:
            grid_search = GridSearchCV(self.clf, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.Y_train)

            # En iyi parametreleri ekrana yazdırma işlemi
            st.write(f'Model: {classifier_name}')
            st.write(f'Best Parameters: {grid_search.best_params_}')

            # Optimum parametrelere göre modeli eğitme işlemi
            self.clf = grid_search.best_estimator_
            self.clf.fit(self.X_train, self.Y_train)

        else:
        # KNN ve SVM dışındaki durumlar için manuel olarak eğitme işlemi
            st.write(f'Model: {classifier_name}')
            self.clf.fit(self.X_train, self.Y_train)

        # Modeli eğitme işlemi
        self.clf.fit(self.X_train, self.Y_train)

    def show_model_results(self):
        # Model sonuçlarını gösterme işlemi
        Y_pred_train = self.clf.predict(self.X_train)
        Y_pred_test = self.clf.predict(self.X_test)

        st.write('Train Set Results:')
        st.write(f'Accuracy: {accuracy_score(self.Y_train, Y_pred_train):.4f}')
        st.write(f'Precision: {precision_score(self.Y_train, Y_pred_train):.4f}')
        st.write(f'Recall: {recall_score(self.Y_train, Y_pred_train):.4f}')
        st.write(f'F1 Score: {f1_score(self.Y_train, Y_pred_train):.4f}')

        st.write('Test Set Results:')
        st.write(f'Accuracy: {accuracy_score(self.Y_test, Y_pred_test):.4f}')
        st.write(f'Precision: {precision_score(self.Y_test, Y_pred_test):.4f}')
        st.write(f'Recall: {recall_score(self.Y_test, Y_pred_test):.4f}')
        st.write(f'F1 Score: {f1_score(self.Y_test, Y_pred_test):.4f}')

        # Confusion Matrix'i gösterme işlemi
        cm_test = confusion_matrix(self.Y_test, Y_pred_test)
        st.write('Confusion Matrix (Test Set):')
        st.write(pd.DataFrame(cm_test, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Test Set)')
        st.pyplot()

        st.write('X_test and Predicted Values:')
        results_df = pd.DataFrame({'Actual Diagnosis': self.Y_test, 'Predicted Diagnosis': Y_pred_test})
        st.write(results_df)
        
        count_of_equal_values = (self.Y_test == Y_pred_test).sum()
        total_tests = len(self.Y_test)
        st.write(f'Total tests: {total_tests} for {self.classifier_name}')
        st.write(f'Count of equal values: {count_of_equal_values} for {self.classifier_name}')


def main():
    st.title('Breast Cancer Diagnosis Prediction')

    app = BreastCancerApp()

    # Bilgisayardan CSV dosyası seçme
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Seçilen dosyayı yükleme
        app.load_data(uploaded_file)

        # Data temizleme ve ön işleme adımları
        app.clean_and_preprocess_data()

        # Model implementasyonu
        classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Naive Bayes'))
        app.implement_model(classifier_name)

        # Model sonuçlarını göster
        app.show_model_results()

if __name__ == '__main__':
    main()
