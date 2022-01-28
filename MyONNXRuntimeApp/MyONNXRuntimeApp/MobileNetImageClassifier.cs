using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyONNXRuntimeApp
{
    public class MobileNetImageClassifier
    {
        const int DimBatchSize = 1;
        const int DimNumberOfChannels = 3;
        const int ImageSizeX = 224;
        const int ImageSizeY = 224;
        const string ModelInputName = "input";
        const string ModelOutputName = "output";

        byte[] _model;
        byte[] _sampleImage;
        List<string> _labels;
        InferenceSession _session;
        Task _initTask;

        public MobileNetImageClassifier()
        {
            _ = InitAsync();
        }

        Task InitAsync()
        {
            if (_initTask == null || _initTask.IsFaulted)
                _initTask = InitTask();

            return _initTask;
        }

        async Task InitTask()
        {
            var assembly = GetType().Assembly;

            // Get labels
            using var labelsStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.imagenet_classes.txt");
            using var reader = new StreamReader(labelsStream);

            string text = await reader.ReadToEndAsync();
            _labels = text.Split(new string[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries).ToList();

            // Get model and create session
            using var modelStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.mobilenetv2-7.onnx");
            using var modelMemoryStream = new MemoryStream();

            modelStream.CopyTo(modelMemoryStream);
            _model = modelMemoryStream.ToArray();
            _session = new InferenceSession(_model);

            // Get sample image
            using var sampleImageStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.SampleImages.dog.png");
            using var sampleImageMemoryStream = new MemoryStream();

            sampleImageStream.CopyTo(sampleImageMemoryStream);
            _sampleImage = sampleImageMemoryStream.ToArray();
        }

        public Task<string> GetClassificationAsync()
        {
            var input = GetImageTensor(_sampleImage);

            using var results = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(ModelInputName, input)
            });

            var output = results.FirstOrDefault(i => i.Name == ModelOutputName);
            var scores = output.AsTensor<float>().ToList();
            var highestScore = scores.Max();
            var highestScoreIndex = scores.IndexOf(highestScore);
            var label = _labels.ElementAt(highestScoreIndex);
            return Task.FromResult(label);
        }

        private DenseTensor<float> GetImageTensor(byte[] image)
        {
            
            using var sourceBitmap = SKBitmap.Decode(image);
            var pixels = sourceBitmap.Bytes;

            //Resize
            if (sourceBitmap.Width != ImageSizeX || sourceBitmap.Height != ImageSizeY)
            {
                float ratio = (float)Math.Min(ImageSizeX, ImageSizeY) / Math.Min(sourceBitmap.Width, sourceBitmap.Height);

                using SKBitmap scaledBitmap = sourceBitmap.Resize(new SKImageInfo(
                    (int)(ratio * sourceBitmap.Width),
                    (int)(ratio * sourceBitmap.Height)),
                    SKFilterQuality.Medium);

                var horizontalCrop = scaledBitmap.Width - ImageSizeX;
                var verticalCrop = scaledBitmap.Height - ImageSizeY;
                var leftOffset = horizontalCrop == 0 ? 0 : horizontalCrop / 2;
                var topOffset = verticalCrop == 0 ? 0 : verticalCrop / 2;

                var cropRect = SKRectI.Create(
                    new SKPointI(leftOffset, topOffset),
                    new SKSizeI(ImageSizeX, ImageSizeY));

                using SKImage currentImage = SKImage.FromBitmap(scaledBitmap);
                using SKImage croppedImage = currentImage.Subset(cropRect);
                using SKBitmap croppedBitmap = SKBitmap.FromImage(croppedImage);

                pixels = croppedBitmap.Bytes;
            }

            //normalize
            var bytesPerPixel = sourceBitmap.BytesPerPixel;
            var rowLength = ImageSizeX * bytesPerPixel;
            var channelLength = ImageSizeX * ImageSizeY;
            var channelData = new float[channelLength * 3];
            var channelDataIndex = 0;

            for (int y = 0; y < ImageSizeY; y++)
            {
                var rowOffset = y * rowLength;

                for (int x = 0, columnOffset = 0; x < ImageSizeX; x++, columnOffset += bytesPerPixel)
                {
                    var pixelOffset = rowOffset + columnOffset;

                    var pixelR = pixels[pixelOffset];
                    var pixelG = pixels[pixelOffset + 1];
                    var pixelB = pixels[pixelOffset + 2];

                    var rChannelIndex = channelDataIndex;
                    var gChannelIndex = channelDataIndex + channelLength;
                    var bChannelIndex = channelDataIndex + (channelLength * 2);

                    channelData[rChannelIndex] = (pixelR / 255f - 0.485f) / 0.229f;
                    channelData[gChannelIndex] = (pixelG / 255f - 0.456f) / 0.224f;
                    channelData[bChannelIndex] = (pixelB / 255f - 0.406f) / 0.225f;

                    channelDataIndex++;
                }
            }

            // create tensor
            var input = new DenseTensor<float>(channelData, new[]
            {
                DimBatchSize,
                DimNumberOfChannels,
                ImageSizeX,
                ImageSizeY
            });

            return input;
        }

    }
}
