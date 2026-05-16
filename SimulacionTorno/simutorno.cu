/*----------------------------------------------------------------------------*/
/*  FICHERO:       simutorno.cu									          */
/*  AUTOR:         Antonio Jimeno											  */
/*													                          */
/*  RESUMEN												                      */
/*  ~~~~~~~												                      */
/* Ejercicio grupal para simulación del movimiento de una herramienta         */
/* tipo torno utilizando GPUs                                                 */
/*----------------------------------------------------------------------------*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>


// includes, project
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simutorno.h"
#include <Windows.h>



#define ERROR_CHECK { cudaError_t err; if ((err = cudaGetLastError()) != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

typedef LARGE_INTEGER timeStamp;
double getTime();

/*----------------------------------------------------------------------------*/
/*  FUNCION A PARALELIZAR  (versión secuencial-CPU)  				          */
/*	Simula el movimiento de una superficie 3D en una máquina tipo torno       */
/*  Realiza pasossim rotaciones de la superficie sobre el eje X de rotacion   */
/*  El giro se da con una definición de PuntosVueltaHelicoide cada 360º       */
/*----------------------------------------------------------------------------*/
int SimulacionTornoCPU(int pasossim, int vtotal, int utotal)
{
	/* Parametros de mecanizado */
	double incA = 360.0 / (double)PuntosVueltaHelicoide;
	for (int u = 0; u<S.UPoints; u++) /* Para cada punto de la superficie */
	{
		for (int v = 0; v<S.VPoints; v++)
		{
			double AvanceMin = 1e10;
            double angle = 0.0;
			for (int i = 0; i < pasossim; i++)  /* Giro del torno en el eje X */
			{
				// Se rota el punto actual (solo interesa la coordenada y)
				double py = S.Buffer[v][u].y * cos(angle*M_PI_180) - S.Buffer[v][u].z * sin(angle*M_PI_180);
				// Calcula la distancia al origen del punto transformado 
				// Si y es la menor se almacena
				if (py<AvanceMin)
				{
					AvanceMin = py; 
				}
				angle += incA;
			}
			int p=S.VPoints*u+v;
			CPUBufferMenorY[p]=AvanceMin;
		}
	}
return OKSIM;
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// FUNCION A IMPLEMENTAR POR EL GRUPO (paralelización de la anterior)
// ---------------------------------------------------------------
// ---------------------------------------------------------------

__global__ void tornoKernel(double* d_Y, double* d_Z, double* d_Result, int pasossim, int total, double incA_rad){
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < total) {

		// PARA AÑADIR EN LA MEMORIA : 
		// [OPTIMIZACIÓN DE MEMORIA] Uso de Registros vs. Memoria Global.
		// La memoria global de la GPU (donde residen d_Y y d_Z) tiene una latencia muy alta.
		// Al volcar estos valores en variables locales ('y', 'z') antes de entrar al bucle,
		// forzamos al compilador a guardarlas en los registros ultra-rápidos del procesador. 
		// Así, los cálculos dentro del 'for' se ejecutan a máxima velocidad.

		double y = d_Y[idx];
		double z = d_Z[idx];

		// Inicializamos el mínimo con un valor arbitrariamente alto ("falso infinito" = 1e10).
		// Esto garantiza que en la primera iteración del bucle, el primer valor 'py' calculado
		// sobreescribirá esta variable, iniciando correctamente la búsqueda del valor mínimo real.

		double AvanceMin = 1e10;
		double angle_rad = 0.0;

		for (int i = 0; i < pasossim; i++)  /* Giro del torno en el eje X */
		{
			// Se rota el punto actual (solo nos interesa la coordenada y)
			// para que añadais en la memoria : 

			// [OPTIMIZACIÓN] Precalculamos el paso angular en radianes en la CPU (Host).
			// Al ser un valor constante para toda la simulación, evitamos que los miles de 
			// hilos de la GPU repitan esta misma multiplicación trigonométrica en cada 
			// iteración del bucle, ahorrando millones de ciclos de reloj innecesarios.

			double py = y * cos(angle_rad) - z * sin(angle_rad);
			// Calcula la distancia al origen del punto transformado 
			// Si y es la menor se almacena
			if (py<AvanceMin)
			{
				AvanceMin = py; 
			}
			angle_rad += incA_rad;
		}

		d_Result[idx] = AvanceMin;
	}




}
 int SimulacionTornoGPU(int pasossim, int vtotal, int utotal)
{

	int total = vtotal * utotal;

	// Lo primero que hacemos es aplanar la malla ( el buffer no es continuo en la memoria )

	double* h_Y = (double*)malloc(total*sizeof(double));
	double* h_Z = (double*)malloc(total*sizeof(double));

	for (int u = 0; u<utotal; u++){
		for (int v = 0; v<vtotal; v++){
			int p = vtotal * u + v;
			h_Y[p] = S.Buffer[v][u].y;
			h_Z[p] = S.Buffer[v][u].z;
		}
	}

	// Reservamos memoria en la GPU para los buffers de entrada y salida

	double* d_Y;
    double* d_Z;
    double* d_Result;
    cudaMalloc((void**)&d_Y,      total * sizeof(double));
    cudaMalloc((void**)&d_Z,      total * sizeof(double));
    cudaMalloc((void**)&d_Result, total * sizeof(double));

	// Copiamos los datos de entrada a la GPU

	cudaMemcpy(d_Y, h_Y, total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z, total * sizeof(double), cudaMemcpyHostToDevice);

	double incA_rad = (360.0 / (double)PuntosVueltaHelicoide) * M_PI_180;

	// lanzamos el kernel

	// Sabemos por cudaGetDeviceProperties que tenemos 22 SMs y 16 bloques máx por SM
	// Lanzamos exactamente eso para ocupar la GPU al 100%
	dim3 block(64);
	dim3 grid(22 * 16);
	tornoKernel<<<grid, block>>>(d_Y, d_Z, d_Result, pasossim, total, incA_rad);

	cudaThreadSynchronize();
    ERROR_CHECK

	// Copiamos el resultado de vuelta a la CPU

	cudaMemcpy(GPUBufferMenorY, d_Result, total * sizeof(double), cudaMemcpyDeviceToHost);

	// Liberamos la memoria de la GPU y CPU

	cudaFree(d_Y);
    cudaFree(d_Z);
    cudaFree(d_Result);
    free(h_Y);
    free(h_Z);

	return OKSIM;
}
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------

 // Declaraciones adelantadas de funciones
 int SimulacionTornoCPU(int pasossim, int vtotal, int utotal);
 int LeerSuperficie(const char *fichero, int& vtotal, int& utotal);



////////////////////////////////////////////////////////////////////////////////
//PROGRAMA PRINCIPAL
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{


	double gpu_start_time, gpu_end_time;
	double cpu_start_time, cpu_end_time;

	/* Numero de argumentos */
	if (argc != 3)
	{
		fprintf(stderr, "Numero de parametros incorecto\n");
		fprintf(stderr, "Uso: %s superficie pasossim\n", argv[0]);
		return;
	}

	/* Datos de los tests */
	int utotal = 0;
	int vtotal = 0;
	/* Apertura de Fichero */
	printf("Prueba simulación torno...\n");
	/* Datos de la superficie */
	if (LeerSuperficie((char *)argv[1], vtotal, utotal) == ERRORSIM)
	{
		fprintf(stderr, "Lectura de superficie incorrecta\n");
		return;
	}
	int pasossim = atoi((char *)argv[2]);
	// Creación buffer resultados para versiones CPU y GPU
	CPUBufferMenorY = (double*)malloc(S.UPoints*S.VPoints*sizeof(double));
	GPUBufferMenorY = (double*)malloc(S.UPoints*S.VPoints*sizeof(double));
	
	/* Algoritmo a paralelizar */
	cpu_start_time = getTime();
	if (SimulacionTornoCPU(pasossim, vtotal, utotal) == ERRORSIM)
	{
		fprintf(stderr, "Simulación CPU incorrecta\n");
		BorrarSuperficie();
		if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
		if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
		exit(1);
	}
	cpu_end_time = getTime();
	/* Algoritmo a implementar */
	cudaSetDevice(0);
	gpu_start_time = getTime();
	if (SimulacionTornoGPU(pasossim, vtotal, utotal) == ERRORSIM)
	{
		fprintf(stderr, "Simulación GPU incorrecta\n");
		BorrarSuperficie();
		if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
		if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
		return;
	}
	cudaDeviceSynchronize();
	gpu_end_time = getTime();
	// Comparación de corrección
	int comprobar = OKSIM;
	for (int i = 0; i<S.UPoints*S.VPoints; i++)
	{
		if (fabs(CPUBufferMenorY[i]-GPUBufferMenorY[i])<1e-6)
		{
			comprobar = ERRORSIM;
			fprintf(stderr, "Fallo en paso %d de simulación, valor correcto Y=%lf\n", i, CPUBufferMenorY[i]);
		}
	}
	// Impresion de resultados
	if (comprobar == OKSIM)
	{
		printf("Simulación correcta!\n");

	}
	// Impresión de resultados
	printf("Tiempo ejecución GPU : %fs\n", \
		gpu_end_time - gpu_start_time);
	printf("Tiempo de ejecución en la CPU : %fs\n", \
		cpu_end_time - cpu_start_time);
	printf("Se ha conseguido un factor de aceleración %fx utilizando CUDA\n", (cpu_end_time - cpu_start_time) / (gpu_end_time - gpu_start_time));
	// Limpieza de buffers
	BorrarSuperficie();
	if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
	if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
	return;
}

int
main(int argc, char** argv)
{
	runTest(argc, argv);
	getchar();
}

/* Funciones auxiliares */
double getTime()
{
	timeStamp start;
	timeStamp dwFreq;
	QueryPerformanceFrequency(&dwFreq);
	QueryPerformanceCounter(&start);
	return double(start.QuadPart) / double(dwFreq.QuadPart);
}



/*----------------------------------------------------------------------------*/
/*	Función:  LeerSuperficie(char *fichero)						              */
/*													                          */
/*	          Lee los datos de la superficie de un fichero con formato .FOR   */
/*----------------------------------------------------------------------------*/
int LeerSuperficie(const char *fichero, int& vtotal, int& utotal)
{
	int i, j, count;		/* Variables de bucle */
	FILE *fpin; 			/* Fichero */
	char cadena[255];
	double x, y, z;

	cadena[0] = 0;
	/* Apertura de Fichero */
	if ((fpin = fopen(fichero, "r")) == NULL) return ERRORSIM;
	/* Lectura de cabecera */
	while (!feof(fpin) && strcmp(cadena, "[HEADER]")) fscanf(fpin, "%s\n", cadena);
	if (fscanf(fpin, "SECTION NUMBER=%d\n", &utotal)<0) return ERRORSIM;
	if (fscanf(fpin, "POINTS PER SECTION=%d\n", &vtotal)<0) return ERRORSIM;
	if (fscanf(fpin, "STEP=%lf\n", &PasoHelicoide)<0) return ERRORSIM;
	if (fscanf(fpin, "POINTS PER ROUND=%d\n", &PuntosVueltaHelicoide)<0) return ERRORSIM;
	if (utotal*vtotal <= 0) return ERRORSIM;
	/* Localizacion de comienzo */
	while (!feof(fpin) && strcmp(cadena, "[GEOMETRY]")) fscanf(fpin, "%s\n", cadena);
	if (feof(fpin)) return ERRORSIM;
	/* Inicialización de parametros geometricos */
	if (CrearSuperficie(utotal, vtotal) == ERRORSIM) return ERRORSIM;
	/* Lectura de coordenadas */
	count = 0;
	for (i = 0; i<utotal; i++)
	{
		for (j = 0; j<vtotal; j++)
		{
			if (!feof(fpin))
			{
				fscanf(fpin, "%lf %lf %lf\n", &x, &y, &z);
				S.Buffer[j][i].x = x;
				S.Buffer[j][i].y = y;
				S.Buffer[j][i].z = z;
				count++;
			}
			else break;
		}
	}
	fclose(fpin);
	if (count != utotal*vtotal) return ERRORSIM;
	return OKSIM;
}



