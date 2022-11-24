#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define Num_0 0
#define Num_1 1
#define Num_2 2
#define Num_point5 0.5
#define for_start 0
#define True 1
#define False 0
#define Array_pos_0 0
#define Array_pos_1 1
#define Array_pos_2 2
#define Array_pos_3 3
#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4
#define lr 0.1f
#define Max_Epoch 10000
#define ErrorHidden 0.0f

double sigmoid(double x) { return Num_1 / (Num_1 + exp(-x)); }
double dSigmoid(double x) { return x * (Num_1 - x); }
double init_weight() { return ((double)rand())/((double)RAND_MAX); }
double loss(double x) { return pow(x,Num_2) * Num_point5;}
void shuffle(int *array, size_t n)
{
    if (n > Num_1)
    {
        size_t i;
        for (i = for_start; i < n - Num_1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + Num_1);
            int t = *(array + j);
            *(array + j) = *(array + i);
            *(array + i) = t;
        }
    }
}
void predict( double* hiddenLayerBias, double** training_inputs, double** hiddenWeights,\
              double* hiddenLayer, double* outputLayerBias, double** outputWeights){

    double* input = (double*)malloc( numInputs * sizeof(double));
    double* TempHLayer = (double*)malloc( numHiddenNodes * sizeof(double));
    int exitFstInput = False, exitSecInput = False;
    double output;

    //ask for input

    while(exitFstInput != True && exitSecInput != True){
        printf("Please enter the first input, or enter '2' to leave.   ");
        scanf("%1lf", &*(input + Array_pos_0));
        if( *(input + Array_pos_0) == Num_2){
            exitFstInput = True;
            break;
        }   
        else if( *(input + Array_pos_0) != Num_0 && *(input + Array_pos_0) != Num_1){
            printf("Please enter '0' or '1' to continue, or '2' to leave.\n");
            continue;
        }

        while(exitSecInput != True){
            printf("Please enter the second input, or enter '2' to leave.   ");
            scanf("%1lf", &*(input + Array_pos_1));
            if( *(input + Array_pos_1) == Num_0 || *(input + Array_pos_1) == Num_1)
                break;
            else if( *(input + Array_pos_1) == Num_2){
                exitFstInput = True;
                exitSecInput = True;
                break;
            }   
            else if( *(input + Array_pos_1) != Num_0 && *(input + Array_pos_1) != Num_1){
                printf("Please enter '0' or '1' to continue, or '2' to leave.\n");
                continue;
            }
        }

        //calculate the result
        
        if( exitFstInput == False && exitSecInput == False){
            for (int j = for_start; j < numHiddenNodes; j++) {
                double activation = *(hiddenLayerBias + j);
                 for (int k = for_start; k<numInputs; k++) {
                    activation += *(input + k) * *( *(hiddenWeights + k) + j);
                }
                *(TempHLayer + j) = sigmoid(activation);
            }
            
            for (int j = for_start; j<numOutputs; j++) {
                double activation = *(outputLayerBias + j);
                for (int k = for_start; k<numHiddenNodes; k++) {
                    activation += *(TempHLayer + k) * *( *(outputWeights + k) + j);
                }
                output = round( sigmoid(activation) );
            }
            printf("The first input:  %d  The second input:  %d  The predict output:  %d\n",\
                    (int)*(input + Array_pos_0), (int)*(input + Array_pos_1), (int)output);
            free(TempHLayer);
        }
    }
    
}

int main(int argc, const char * argv[]) {
    
    double* hiddenLayer = (double*)malloc( numHiddenNodes * sizeof(double) );
    double* outputLayer = (double*)malloc( numOutputs * sizeof(double) );
    
    double* hiddenLayerBias = (double*)malloc( numHiddenNodes * sizeof(double) );
    double* outputLayerBias = (double*)malloc( numOutputs * sizeof(double) );

    double** hiddenWeights = (double**)malloc( numInputs * sizeof(double*) );
    for (int i = for_start; i < numInputs; i++)
        *(hiddenWeights + i) = (double*)malloc( numHiddenNodes * sizeof(double) );

    double** outputWeights = (double**)malloc( numHiddenNodes * sizeof(double*) );
    for (int i = for_start; i < numHiddenNodes; i++)
        *(outputWeights + i) = (double*)malloc( numOutputs * sizeof(double) );
    
    double** training_inputs = (double**)malloc( numTrainingSets * sizeof(double*) );
    for (int i = for_start; i < numTrainingSets; i++)
        *(training_inputs + i) = (double*)malloc( numInputs * sizeof(double) );

    *( *(training_inputs + Array_pos_0) + Array_pos_0) = Num_0;
    *( *(training_inputs + Array_pos_0) + Array_pos_1) = Num_0;
    *( *(training_inputs + Array_pos_1) + Array_pos_0) = Num_0;
    *( *(training_inputs + Array_pos_1) + Array_pos_1) = Num_1;
    *( *(training_inputs + Array_pos_2) + Array_pos_0) = Num_1;
    *( *(training_inputs + Array_pos_2) + Array_pos_1) = Num_0;
    *( *(training_inputs + Array_pos_3) + Array_pos_0) = Num_1;
    *( *(training_inputs + Array_pos_3) + Array_pos_1) = Num_1;

    double** training_outputs = (double**)malloc( numTrainingSets * sizeof(double*) );
    for (int i = for_start; i < numTrainingSets; i++)
        *(training_outputs + i) = (double*)malloc( numOutputs * sizeof(double) );

    *( *(training_outputs + Array_pos_0) + Array_pos_0) = Num_0;
    *( *(training_outputs + Array_pos_1) + Array_pos_0) = Num_1;
    *( *(training_outputs + Array_pos_2) + Array_pos_0) = Num_1;
    *( *(training_outputs + Array_pos_3) + Array_pos_0) = Num_0;

    FILE * fp;
    fp = fopen ("file.txt", "w");
    
    for (int i = for_start; i<numInputs; i++) {
        for (int j = for_start; j<numHiddenNodes; j++) {
            *( *(hiddenWeights + i) + j) = init_weight();
        }
    }
    for (int i = for_start; i<numHiddenNodes; i++) {
        *(hiddenLayerBias + i) = init_weight();
        for (int j = for_start; j<numOutputs; j++) {
            *( *(outputWeights + i) + j) = init_weight();
        }
    }
    for (int i = for_start; i<numOutputs; i++) {
        *(outputLayerBias + i) = init_weight();
    }
    
    int* trainingSetOrder = (int*)malloc( numTrainingSets * sizeof(int));
    for (int i = for_start; i < numTrainingSets; i++)
        *(trainingSetOrder + i) = i;
    
    for (int n = for_start; n < Max_Epoch; n++) {
        shuffle(trainingSetOrder,numTrainingSets);
        for (int x = for_start; x<numTrainingSets; x++) {
            
            int i = *(trainingSetOrder + x);
            
            // Forward pass
            
            for (int j = for_start; j<numHiddenNodes; j++) {
                double activation = *(hiddenLayerBias + j);
                 for (int k = for_start; k<numInputs; k++) {
                    activation += *( *(training_inputs + i) + k) * *( *(hiddenWeights + k) + j);
                }
                *(hiddenLayer + j) = sigmoid(activation);
            }
            
            for (int j = for_start; j<numOutputs; j++) {
                double activation = *(outputLayerBias + j);
                for (int k = for_start; k<numHiddenNodes; k++) {
                    activation += *(hiddenLayer + k) * *( *(outputWeights + k) + j);
                }
                *(outputLayer + j) = sigmoid(activation);
            }
            
            printf( "Epoch: %5d    Input: %f %f    Output: %f    Expected Output: %f    ",\
                    n + Num_1, *( *(training_inputs + i) + Array_pos_0), *( *(training_inputs + i) + Array_pos_1), *(outputLayer + Array_pos_0), *( *(training_outputs + i) + Array_pos_0));

           // Backprop
            
            double* deltaOutput = (double*)malloc(numOutputs * sizeof(double));
            for (int j = for_start; j<numOutputs; j++) {
                double errorOutput = ( *( *(training_outputs + i) + j) - *(outputLayer + j));
                *(deltaOutput + j) = errorOutput * dSigmoid( *(outputLayer + j));
                double LOSS = loss(errorOutput);
                printf("Loss: %f\n", LOSS);
    
                fprintf(fp, "%f\n", LOSS);
            }
            
      
            double* deltaHidden = (double*)malloc(numHiddenNodes * sizeof(double));
            for (int j = for_start; j<numHiddenNodes; j++) {
                double errorHidden = ErrorHidden;
                for(int k = for_start; k<numOutputs; k++) {
                    errorHidden += *(deltaOutput + k) * *( *(outputWeights + j) + k);
                }
                *(deltaHidden + j) = errorHidden*dSigmoid( *(hiddenLayer + j));
            }
            
            for (int j = for_start; j<numOutputs; j++) {
                *(outputLayerBias + j) += *(deltaOutput + j) * lr;
                for (int k = for_start; k<numHiddenNodes; k++) {
                    *( *(outputWeights + k) + j) += *(hiddenLayer + k) * *(deltaOutput + j) * lr;
                }
            }
            
            for (int j = for_start; j<numHiddenNodes; j++) {
                *(hiddenLayerBias + j) += *(deltaHidden + j) * lr;
                for(int k = for_start; k<numInputs; k++) {
                    *( *(hiddenWeights + k) + j) += *( *(training_inputs + i) + k) * *(deltaHidden + j) * lr;
                }
            }
        }
    }
    
    // Print weights
    printf("Final Hidden Weights\n[ ");
    for (int j = for_start; j<numHiddenNodes; j++) {
        printf("[ ");
        for(int k = for_start; k<numInputs; k++) {
            printf("%f ", *( *(hiddenWeights + k) + j));
        }
        printf("] ");
    }
    printf("]\n");
    
    printf("Final Hidden Biases\n[ ");
    for (int j = for_start; j<numHiddenNodes; j++) {
        printf("%f ", *(hiddenLayerBias + j));
    }
    printf("]\n");
    printf("Final Output Weights");
    for (int j = for_start; j<numOutputs; j++) {
        printf("[ ");
        for (int k = for_start; k<numHiddenNodes; k++) {
            printf("%f ", *( *(outputWeights + k) + j));
        }
        printf("]\n");
    }
    printf("Final Output Biases\n[ ");
    for (int j = for_start; j<numOutputs; j++) {
        printf("%f ", *(outputLayerBias + j));
    }
    printf("]\n");

    predict( hiddenLayerBias, training_inputs, hiddenWeights, hiddenLayer, outputLayerBias, outputWeights);
    fclose(fp);
    return Num_0;
}