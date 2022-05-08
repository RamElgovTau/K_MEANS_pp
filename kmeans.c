#include <python.h>
#include <stdio.h>
#include<math.h>
#include <stdlib.h>
#include <string.h>
/*********************************
ram elgov 206867517
mohammad daghash 314811290
**********************************/
int is_converged(double *centroids, double *old_centroids, int K, int vector_length) {
  int i, j;
  double norm;
  /*
   * checks if all centroid's norm didn't change more than the value of epsilon. if yes, then it's converged.
   */
  for (i = 0; i < K; i++) {
    norm = 0;
    for (j = 0; j < vector_length; j++) {
      norm += pow(centroids[i * vector_length + j] - old_centroids[i * vector_length + j], 2);
    }
    norm = pow(norm, 0.5);
    if (norm >= 0.001) {
      return 1;
    }
  }
  return 0;
}
int indexof_closest_cluster(double *x, double *centroids, int K, int vector_length) {
  double min = 0;
  double sum;
  int i, j, index = 0;
  /* minimum initialisation. (the fis  */
  for (i = 0; i < vector_length; i++) {
    min += pow(x[i] - centroids[i], 2);
  }
  /* checks for the rest of the centroids. */
  for (j = 0; j < K; j++) {
    sum = 0;
    for (i = 0; i < vector_length; i++) {
      sum += pow(x[i] - centroids[j * vector_length + i], 2);
    }
    if (sum < min) {
      min = sum;
      index = j;
    }
  }
  return index;
}

int main(int argc, char **argv) {
  FILE *ifp, *ofp;
  double vec = 0;
  char c;
  int file_length = 0, vector_length = 0, iteration_num;
  int valid;
  int i = 0, j;
  int maxiter, K, digit_num, counter;
  double *vectors;
  double **data_points;
  double *centroids;
  double *old_centroids;
  double *clusters;
  int *sizeof_clusters;
  if (argc == 5) {
    K = atoi(argv[1]);
    maxiter = atoi(argv[2]);
    ifp = fopen(argv[3], "r");
    ofp = fopen(argv[4], "w");
  } else {
    K = atoi(argv[1]);
    maxiter = 200;
    ifp = fopen(argv[2], "r");
    ofp = fopen(argv[3], "w");
  }
  if(K<=1 || maxiter<=0){
    printf("Invalid Input!");
    return 1;
  }
  while ((c = fgetc(ifp)) != EOF) {
    if (c == '\n') {
      file_length++;
      vector_length++;
    }
    if (c == ',') {
      vector_length++;
    }
  }
  vector_length = vector_length / file_length;
  if(K>=file_length){
    printf("Invalid Input!");
    return 1;
  }
  vectors = calloc(vector_length * file_length, sizeof(double));
  data_points = calloc(file_length, sizeof(double *));
  centroids = calloc(K * vector_length, sizeof(double));
  old_centroids = calloc(K * vector_length, sizeof(double));
  clusters = calloc(K * vector_length, sizeof(double));
  sizeof_clusters = calloc(K, sizeof(int));
  rewind(ifp);
  while (fscanf(ifp, "%lf,", &vec) != EOF) {
    vectors[i] = vec;
    i++;
  }
  for (i = 0; i < file_length; i++) {
    data_points[i] = calloc(vector_length, sizeof(double));
  }
  for (i = 0; i < file_length; i++) {
    for (j = 0; j < vector_length; j++) {
      data_points[i][j] = vectors[i * vector_length + j];
    }
  }
  for (i = 0; i < vector_length * K; i++) {
    centroids[i] = vectors[i];
  }
  iteration_num = 0;
  valid = 1;
  while (iteration_num < maxiter && valid == 1) {
    for (i = 0; i < K * vector_length; i++) {
      old_centroids[i] = centroids[i];
    }
    for (i = 0; i < file_length; i++) {
      int index = indexof_closest_cluster(data_points[i], centroids, K, vector_length);
      for (j = 0; j < vector_length; j++) {
        clusters[index * vector_length + j] += data_points[i][j];
      }
      sizeof_clusters[index]++;
    }
    for (j = 0; j < K; j++) {
      for (i = 0; i < vector_length; i++) {
        centroids[vector_length * j + i] = clusters[vector_length * j + i] / sizeof_clusters[j];
      }
    }
    for (j = 0; j < K * vector_length; j++) {
      clusters[j] = 0;
    }
    for (j = 0; j < K; j++) {
      sizeof_clusters[j] = 0;
    }
    valid = is_converged(centroids, old_centroids, K, vector_length);
    iteration_num++;
  }
  for (i = 0; i < K; i++) {
    for (j = 0; j < vector_length; j++) {
      fprintf(ofp, "%.4f", centroids[i * vector_length + j]);
      if (j < vector_length - 1) {
        fprintf(ofp, ",");
      } else {
        fprintf(ofp, "\n");
      }
    }
  }
  counter=0 ;
  digit_num=strlen(argv[1]);
  for(i=0;i<digit_num;i++){
    if((((int)(argv[1][i])-48)<=9) && (((int)(argv[1][i])-48)>=0)){
      counter++;
    }
  }
  if(digit_num!=counter){
    printf("Invalid Input!");
    return 1;
  }
  counter=0 ;
  if(argc==5){
    digit_num=strlen(argv[2]);
    for(i=0;i<digit_num;i++){
      if((((int)(argv[2][i])-48)<=9) && (((int)(argv[2][i])-48)>=0)){
        counter++;
      }
    }
    if(digit_num!=counter){
      printf("Invalid Input!");
      return 1;
    }
  }
  free(clusters);
  free(sizeof_clusters);
  free(centroids);
  free(old_centroids);
  free(vectors);
  for (i = 0; i < file_length; i++) {
    free(data_points[i]);
  }
  free(data_points);
  fclose(ifp);
  fclose(ofp);
  return 0;
}
