#include <stdio.h>
#include "MRFEnergy.h"
#include <iostream>
#include <fstream>
#include "ordering.cpp"
#include "minimize.cpp"
#include "treeprobabilities.cpp"
#include "instances.inc"
#include "NumCpp.hpp"

// Example: minimizing an energy function with Potts terms.
// See type*.h files for other types of terms.

void calculate(std::string corr_bin, std::string disp_bin, std::string save_bin, bool syn = true)
{

	//image_size 540*629 or 550*700 labels 10/20 
	auto dataCorr = nc::load<float>(corr_bin);
	dataCorr.reshape({ 540 * 629, 20 });

	auto dataDisp = nc::load<int64_t>(disp_bin); //float  int64_t
	dataDisp.reshape({ 540 * 629, 20 });

	//std::cout << dataDisp(100, 3) << std::endl;
	//std::cout << dataCorr(100, 3) << std::endl;

	const int imRows = 540;
	const int imCols = 629;
	const int numLabels = 20;
	const int nodeNum = imRows * imCols; // number of nodes


	MRFEnergy<TypeGeneral>* mrf;
	MRFEnergy<TypeGeneral>::NodeId* nodes;
	MRFEnergy<TypeGeneral>::Options options;
	TypeGeneral::REAL energy, lowerBound;


	mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
	nodes = new MRFEnergy<TypeGeneral>::NodeId[nodeNum];

	float lambda_d = 1.0;
	float lambda_s = 200.0;

	//int nodeIndex[imRows][imCols];
	//int resultDisp[imRows][imCols];

	int** nodeIndex = new int* [imRows];
	for (int i = 0; i < imRows; i++)
		nodeIndex[i] = new int[imCols];

	int** resultDisp = new int* [imRows];
	for (int i = 0; i < imRows; i++)
		resultDisp[i] = new int[imCols];

	TypeGeneral::REAL V[20 * 20];
	TypeGeneral::REAL D[20];

	int r, c, l1, l2, i, rl, id;

	//
	int curId = 0;
	int tmpFail = 0;
	int tmpF = 0;
	for (r = 0; r < imRows; r++)
	{
		for (c = 0; c < imCols; c++)
		{
			tmpF = 0;
			nodeIndex[r][c] = curId;
			//printf("%d\n", nodeIndex[r][c]);

			for (i = 0; i < numLabels; i++)
			{
				D[i] = dataCorr(curId, i) * lambda_d;
				if (dataDisp(curId, i) <= 0.0 || dataDisp(curId, i) >= 101.0)
				{
					tmpF++;
					D[i] = 2000.0;
				}
				if (tmpF >= 20)
					tmpFail++;
			}

			nodes[curId] = mrf->AddNode(TypeGeneral::LocalSize(numLabels), TypeGeneral::NodeData(D));
			curId++;
		}
	}
	printf("%d\n", tmpFail);
	int id1;
	int id2;
	for (r = 0; r < imRows - 1; r++) //imRows - 1
	{
		for (c = 0; c < imCols; c++) //imCols
		{
			id1 = nodeIndex[r][c];
			id2 = nodeIndex[r + 1][c];

			//printf("%d\n", nodeIndex[r][c]);
			for (l1 = 0; l1 < numLabels; l1++)

			{
				for (l2 = 0; l2 < numLabels; l2++)
				{
					//V[l1 + l2 * numLabels] = 0;

					if (abs(dataDisp(id1, l1) - dataDisp(id2, l2)) <= 3.0)
						V[l1 + l2 * numLabels] = 0.0;
					else
						V[l1 + l2 * numLabels] = 1.0 * lambda_s;
				}
			}

			mrf->AddEdge(nodes[id1], nodes[id2], TypeGeneral::EdgeData(TypeGeneral::GENERAL, V));
		}
	}

	for (r = 0; r < imRows; r++)
	{
		for (c = 0; c < imCols - 1; c++)
		{
			id1 = nodeIndex[r][c];
			id2 = nodeIndex[r][c + 1];

			for (l1 = 0; l1 < numLabels; l1++)
				for (l2 = 0; l2 < numLabels; l2++)
				{
					if (abs(dataDisp(id1, l1) - dataDisp(id2, l2)) <= 3.0)
						V[l1 + l2 * numLabels] = 0;
					else
						V[l1 + l2 * numLabels] = 1.0 * lambda_s;
				}
			mrf->AddEdge(nodes[id1], nodes[id2], TypeGeneral::EdgeData(TypeGeneral::GENERAL, V));
		}
	}

	// Function below is optional - it may help if, for example, nodes are added in a random order
	// mrf->SetAutomaticOrdering();

	/////////////////////// TRW-S algorithm //////////////////////
	options.m_iterMax = 200; // maximum number of iterations
	mrf->Minimize_TRW_S(options, lowerBound, energy);

	int tmp = 0;
	for (r = 0; r < imRows; r++)
		for (c = 0; c < imCols; c++)
		{
			id = nodeIndex[r][c];
			rl = mrf->GetSolution(nodes[id]);
			resultDisp[r][c] = int(dataDisp(id, rl));

			if (resultDisp[r][c] == 0)
				tmp++;
		}
	std::cout << tmp << std::endl;
	//std::cout << resultDisp[300][300] << std::endl;

	std::fstream myFile;
	myFile.open(save_bin, std::ios::out | std::ios::binary);
	for (int i = 0; i < imRows; i++)
		for (int j = 0; j < imCols; j++)
			myFile.write((char*)&resultDisp[i][j], sizeof(int));

	std::cout << "save current results" << std::endl;

	// done
	delete nodes;
	delete mrf;
}


void main()
{

	std::string base_folder = "data_uncode_npy_20/";
	std::string folder_idx[1] = { "0"};
	std::string para_idx[1] = {"0.1"};

	for (int i = 0; i <1 ; i++)
		for (int j = 0; j < 1; j++)
		{
			std::string corr_bin = base_folder + folder_idx[i] + "/corr/" + para_idx[j] + ".bin";
			std::string disp_bin = base_folder + folder_idx[i] + "/disp/" + para_idx[j] + ".bin";
			std::string save_bin = base_folder + folder_idx[i] + "/results/" + para_idx[j] + ".bin";

			calculate(corr_bin, disp_bin, save_bin);
			std::cout << corr_bin << std::endl;
			std::cout << disp_bin << std::endl;
		}

}
