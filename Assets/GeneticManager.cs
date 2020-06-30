using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using TMPro;

public class GeneticManager : MonoBehaviour
{
    [Header("References")]
    public CarController controller;

    [Header("Mutacion")]
    public int Poblacion = 85;
    [Range(0.0f, 1.0f)]
    public float ProbabilidadMutacion = 0.055f;

    [Header("Padres e hijos")]
    public int Mejores = 8;
    public int Peores = 3;
    public int Hijos;

    private List<int> genePool = new List<int>();

    private int naturallySelected;

    private NNet[] population;

    [Header("Generaciones")]
    public int currentGeneration;
    public int currentGenome = 0;


    private void CreatePopulation()
    {
        population = new NNet[Poblacion];
        FillPopulationWithRandomValues(population, 0);
        ResetToCurrentGenome();
    }

    private void ResetToCurrentGenome()
    {
        controller.ResetWithNetwork(population[currentGenome]);
    }

    private void FillPopulationWithRandomValues(NNet[] newPopulation, int startingIndex)
    {
        while(startingIndex < Poblacion)
        {
            newPopulation[startingIndex] = new NNet();
            newPopulation[startingIndex].Initialise(controller.LAYERS, controller.NEURONS);
            startingIndex++;
        }
    }

    public void Death(float fitness, NNet network)
    {
        if (currentGenome < population.Length -1)
        {
            population[currentGenome].fitness = fitness;
            currentGenome++;
            ResetToCurrentGenome();
        }
        else
        {
            population[currentGenome].fitness = fitness;
            currentGenome++;
            RePopulate();
        }
    }

    private void RePopulate()
    {
        genePool.Clear();
        currentGeneration++;
        naturallySelected = 0;
        Sort();

        NNet[] newPopulation = Pick();

        Crossover(newPopulation);
        Mutate(newPopulation);

        FillPopulationWithRandomValues(newPopulation, naturallySelected);

        population = newPopulation;
        currentGenome = 0;
        ResetToCurrentGenome();
    }

    private void Mutate(NNet[] newPopulation)
    {
        for (int i = 0; i < naturallySelected; i++)
        {
            for (int j = 0; j < newPopulation[i].weights.Count; j++)
            {
                if (Random.Range(0.0f, 1.0f) < ProbabilidadMutacion)
                {
                    newPopulation[i].weights[j] = MutateMatrix(newPopulation[i].weights[j]);
                }
            }
        }
    }

    private Matrix<float> MutateMatrix(Matrix<float> a)
    {
        int randomPoints = Random.Range(1, (a.RowCount * a.ColumnCount) / 7);

        Matrix<float> c = a;

        for (int i = 0; i < randomPoints; i++)
        {
            int randomColumn = Random.Range(0, c.ColumnCount);
            int randomRow = Random.Range(0, c.RowCount);

            c[randomRow, randomColumn] = Mathf.Clamp(c[randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
        }

        return c;
    }

    private void Crossover(NNet[] newPopulation)
    {
        for (int i = 0; i < Hijos; i += 2)
        {
            int AIndex = i;
            int BIndex = i + 1;

            if (genePool.Count >= 1)
            {
                for (int j = 0; j < 100; j++)
                {
                    AIndex = genePool[Random.Range(0, genePool.Count)];
                    BIndex = genePool[Random.Range(0, genePool.Count)];

                    if (AIndex != BIndex)
                    {
                        break;
                    }
                }
            }

            NNet child1 = new NNet();
            NNet child2 = new NNet();

            child1.Initialise(controller.LAYERS, controller.NEURONS);
            child2.Initialise(controller.LAYERS, controller.NEURONS);

            child1.fitness = 0;
            child2.fitness = 0;

            for (int k = 0; k < child1.weights.Count; k++)
            {
                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    child1.weights[k] = population[AIndex].weights[k];
                    child2.weights[k] = population[BIndex].weights[k];
                }
                else
                {
                    child1.weights[k] = population[BIndex].weights[k];
                    child2.weights[k] = population[AIndex].weights[k];
                }
            }

            for (int k = 0; k < child1.biases.Count; k++)
            {
                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    child1.biases[k] = population[AIndex].biases[k];
                    child2.biases[k] = population[BIndex].biases[k];
                }
                else
                {
                    child1.biases[k] = population[BIndex].biases[k];
                    child2.biases[k] = population[AIndex].biases[k];
                }
            }

            newPopulation[naturallySelected] = child1;
            naturallySelected++;
            newPopulation[naturallySelected] = child2;
            naturallySelected++;
        }
    }

    private NNet[] Pick()
    {
        NNet[] newPopulation = new NNet[Poblacion];

        for(int i = 0; i < Mejores; i++)
        {
            newPopulation[naturallySelected] = population[i].InitialiseCopy(controller.LAYERS, controller.NEURONS);
            newPopulation[naturallySelected].fitness = 0;
            naturallySelected++;

            int f = Mathf.RoundToInt(population[i].fitness * 10);

            for (int j = 0; j < f; j++)
            {
                genePool.Add(i);
            }
        }
        for (int i = 0; i < Peores; i++)
        {
            int last = population.Length - 1;
            last -= i;

            int f = Mathf.RoundToInt(population[last].fitness * 10);

            for (int j = 0; j < f; j++)
            {
                genePool.Add(last);
            }
        }
        return newPopulation;
    }

    private void Sort()
    {
        for (int i = 0; i < population.Length; i++)
        {
            for (int j = i; j < population.Length; j++)
            {
                if (population[i].fitness < population[j].fitness)
                {
                    NNet aux = population[i];
                    population[i] = population[j];
                    population[j] = aux;
                }
            }
        }
    }



    // Start is called before the first frame update
    void Start()
    {
        CreatePopulation();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
