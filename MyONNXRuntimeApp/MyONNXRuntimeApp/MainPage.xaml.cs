using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Forms;

namespace MyONNXRuntimeApp
{
    public partial class MainPage : ContentPage
    {
        MobileNetImageClassifier _classifier = new MobileNetImageClassifier();
        public MainPage()
        {
            InitializeComponent();
        }

        async void OnButtonClicked(object sender, EventArgs args)
        {
            var result = await _classifier.GetClassificationAsync();
            await DisplayAlert("Result", result, "OK");
        }
    }
}
