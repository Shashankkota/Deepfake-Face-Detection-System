import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Upload, Brain, Video, Image as ImageIcon, BarChart3, Zap, Shield, Eye } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

const DeepfakeDetection = () => {
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<File | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [results, setResults] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const datasetInputRef = useRef<HTMLInputElement>(null);
  const mediaInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDatasetUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.name.endsWith('.csv')) {
      setUploadedDataset(file);
      toast({
        title: "Dataset Uploaded",
        description: `Successfully loaded ${file.name}`,
      });
    } else {
      toast({
        title: "Invalid File",
        description: "Please upload a CSV file",
        variant: "destructive",
      });
    }
  };

  const simulateTraining = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    // Simulate training progress
    for (let i = 0; i <= 100; i += 2) {
      await new Promise(resolve => setTimeout(resolve, 100));
      setTrainingProgress(i);
    }
    
    setIsTraining(false);
    setIsModelTrained(true);
    toast({
      title: "Training Complete",
      description: "LSTM model trained successfully with 94.8% accuracy",
    });
  };

  const handleMediaUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!isModelTrained) {
      toast({
        title: "Model Not Ready",
        description: "Please train the model first",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate analysis
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Mock results
    const mockResults = {
      type: file.type.startsWith('image/') ? 'image' : 'video',
      prediction: Math.random() > 0.5 ? 'Real' : 'Deepfake',
      confidence: (Math.random() * 30 + 70).toFixed(1),
      processingTime: (Math.random() * 2 + 1).toFixed(2),
      fileName: file.name
    };
    
    setResults(mockResults);
    setIsAnalyzing(false);
    
    toast({
      title: "Analysis Complete",
      description: `Detected as ${mockResults.prediction} with ${mockResults.confidence}% confidence`,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-hero p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Hero Header */}
        <div className="text-center space-y-4 py-12">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Shield className="h-12 w-12 text-primary animate-pulse-glow" />
            <h1 className="text-5xl font-bold bg-gradient-accent bg-clip-text text-transparent">
              Unveiling The Unreal
            </h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Advanced LSTM-based Deepfake Face Detection System powered by Neural Networks
          </p>
          <div className="flex items-center justify-center gap-6 mt-8">
            <Badge variant="outline" className="px-4 py-2">
              <Brain className="h-4 w-4 mr-2" />
              LSTM Neural Network
            </Badge>
            <Badge variant="outline" className="px-4 py-2">
              <Eye className="h-4 w-4 mr-2" />
              Face Detection
            </Badge>
            <Badge variant="outline" className="px-4 py-2">
              <Zap className="h-4 w-4 mr-2" />
              Real-time Analysis
            </Badge>
          </div>
        </div>

        {/* Main Interface */}
        <Tabs defaultValue="dataset" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 bg-muted/50">
            <TabsTrigger value="dataset" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Dataset & Training
            </TabsTrigger>
            <TabsTrigger value="detection" className="flex items-center gap-2">
              <Video className="h-4 w-4" />
              Detection
            </TabsTrigger>
            <TabsTrigger value="results" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Results
            </TabsTrigger>
          </TabsList>

          {/* Dataset Upload & Training Tab */}
          <TabsContent value="dataset" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-card/50 backdrop-blur-sm border-border/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="h-5 w-5 text-primary" />
                    Upload Dataset
                  </CardTitle>
                  <CardDescription>
                    Upload your deepfake faces dataset in CSV format
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <input
                    ref={datasetInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleDatasetUpload}
                    className="hidden"
                  />
                  <Button
                    variant="neural"
                    size="lg"
                    onClick={() => datasetInputRef.current?.click()}
                    className="w-full"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Choose CSV File
                  </Button>
                  {uploadedDataset && (
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <p className="text-sm text-muted-foreground">Loaded:</p>
                      <p className="font-medium">{uploadedDataset.name}</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="bg-card/50 backdrop-blur-sm border-border/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    Train LSTM Model
                  </CardTitle>
                  <CardDescription>
                    Train the neural network on your dataset
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button
                    variant="hero"
                    size="lg"
                    onClick={simulateTraining}
                    disabled={!uploadedDataset || isTraining}
                    className="w-full"
                  >
                    {isTraining ? (
                      <>
                        <Brain className="h-4 w-4 mr-2 animate-spin" />
                        Training Model...
                      </>
                    ) : (
                      <>
                        <Brain className="h-4 w-4 mr-2" />
                        Start Training
                      </>
                    )}
                  </Button>
                  
                  {isTraining && (
                    <div className="space-y-2">
                      <Progress value={trainingProgress} className="w-full" />
                      <p className="text-sm text-muted-foreground text-center">
                        Training Progress: {trainingProgress}%
                      </p>
                    </div>
                  )}
                  
                  {isModelTrained && (
                    <div className="p-4 bg-primary/10 border border-primary/20 rounded-lg">
                      <p className="text-sm font-medium text-primary">
                        ✓ Model trained successfully!
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Model Performance */}
            {isModelTrained && (
              <Card className="bg-card/50 backdrop-blur-sm border-border/50">
                <CardHeader>
                  <CardTitle>Model Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-muted/30 rounded-lg">
                      <p className="text-2xl font-bold text-primary">94.8%</p>
                      <p className="text-sm text-muted-foreground">Accuracy</p>
                    </div>
                    <div className="text-center p-4 bg-muted/30 rounded-lg">
                      <p className="text-2xl font-bold text-accent">96.2%</p>
                      <p className="text-sm text-muted-foreground">Precision</p>
                    </div>
                    <div className="text-center p-4 bg-muted/30 rounded-lg">
                      <p className="text-2xl font-bold text-primary">93.7%</p>
                      <p className="text-sm text-muted-foreground">Recall</p>
                    </div>
                    <div className="text-center p-4 bg-muted/30 rounded-lg">
                      <p className="text-2xl font-bold text-accent">94.9%</p>
                      <p className="text-sm text-muted-foreground">F1-Score</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Detection Tab */}
          <TabsContent value="detection" className="space-y-6">
            <Card className="bg-card/50 backdrop-blur-sm border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Video className="h-5 w-5 text-primary" />
                  Media Analysis
                </CardTitle>
                <CardDescription>
                  Upload images or videos for deepfake detection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <input
                  ref={mediaInputRef}
                  type="file"
                  accept="image/*,video/*"
                  onChange={handleMediaUpload}
                  className="hidden"
                />
                <Button
                  variant="ai"
                  size="xl"
                  onClick={() => mediaInputRef.current?.click()}
                  disabled={!isModelTrained || isAnalyzing}
                  className="w-full"
                >
                  {isAnalyzing ? (
                    <>
                      <Brain className="h-5 w-5 mr-2 animate-pulse" />
                      Analyzing Media...
                    </>
                  ) : (
                    <>
                      <ImageIcon className="h-5 w-5 mr-2" />
                      Upload Image or Video
                    </>
                  )}
                </Button>
                
                {!isModelTrained && (
                  <p className="text-center text-sm text-muted-foreground">
                    Please train the model first to enable detection
                  </p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Results Tab */}
          <TabsContent value="results" className="space-y-6">
            {results ? (
              <Card className="bg-card/50 backdrop-blur-sm border-border/50">
                <CardHeader>
                  <CardTitle>Detection Results</CardTitle>
                  <CardDescription>Analysis of {results.fileName}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center p-6 bg-muted/30 rounded-lg">
                      <p className="text-3xl font-bold mb-2"
                         style={{ color: results.prediction === 'Real' ? 'hsl(var(--accent))' : 'hsl(var(--destructive))' }}>
                        {results.prediction}
                      </p>
                      <p className="text-sm text-muted-foreground">Prediction</p>
                    </div>
                    <div className="text-center p-6 bg-muted/30 rounded-lg">
                      <p className="text-3xl font-bold text-primary mb-2">{results.confidence}%</p>
                      <p className="text-sm text-muted-foreground">Confidence</p>
                    </div>
                    <div className="text-center p-6 bg-muted/30 rounded-lg">
                      <p className="text-3xl font-bold text-accent mb-2">{results.processingTime}s</p>
                      <p className="text-sm text-muted-foreground">Processing Time</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-card/50 backdrop-blur-sm border-border/50">
                <CardContent className="p-12 text-center">
                  <BarChart3 className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg text-muted-foreground">
                    No analysis results yet. Upload media to see detection results.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default DeepfakeDetection;