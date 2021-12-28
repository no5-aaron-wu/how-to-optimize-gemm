#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // avx

/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Block sizes */
#define mc 120
#define kc 128
#define nb 1000

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot4x4( int, double *, int, double *, int, double *, int );
void AddDot12x4( int, double *, int, double *, int, double *, int );
void PackMatrixA( int, double *, int, double * );
void PackMatrixA12x4( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, int, double *, int, int );
void InnerKernel12x4( int, int, int, double *, int, double *, int, double *, int, int );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, p, pb, ib;

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */

  for ( p=0; p<k; p+=kc ){
    pb = min( k-p, kc );
    for ( i=0; i<m; i+=mc ){
      ib = min( m-i, mc );
      InnerKernel12x4( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc, i==0 );
    }
  }
}

void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  int i, j;
  double 
    packedA[ m * k ];
  static double 
    packedB[ kc*nb ];    /* Note: using a static buffer is not thread safe... */

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    if ( first_time )
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if ( j == 0 ) 
	PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
      AddDot4x4( k, &packedA[ i*k ], 4, &packedB[ j*k ], k, &C( i,j ), ldc );
    }
  }
}

void InnerKernel12x4( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  int i, j;
  double 
    packedA[ m * k ];

  static double 
    packedB[ kc*nb ];    /* Note: using a static buffer is not thread safe... */

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    if ( first_time )
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
    for ( i=0; i<m; i+=12 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if ( j == 0 ) {
	      PackMatrixA12x4( k, &A( i, 0 ), lda, &packedA[ i*k ] );
      }
      AddDot12x4( k, &packedA[ i*k ], 4, &packedB[ j*k ], k, &C( i,j ), ldc );
    }
  }
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    double 
      *a_ij_pntr = &A( 0, j );

    *a_to     = *a_ij_pntr;
    *(a_to+1) = *(a_ij_pntr+1);
    *(a_to+2) = *(a_ij_pntr+2);
    *(a_to+3) = *(a_ij_pntr+3);

    a_to += 4;
  }
}

void PackMatrixA12x4( int k, double *a, int lda, double *a_to )
{
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    double 
      *a_ij_pntr = &A( 0, j );

    *a_to     = *a_ij_pntr;
    *(a_to+1) = *(a_ij_pntr+1);
    *(a_to+2) = *(a_ij_pntr+2);
    *(a_to+3) = *(a_ij_pntr+3);
    *(a_to+4) = *(a_ij_pntr+4);
    *(a_to+5) = *(a_ij_pntr+5);
    *(a_to+6) = *(a_ij_pntr+6);
    *(a_to+7) = *(a_ij_pntr+7);
    *(a_to+8) = *(a_ij_pntr+8);
    *(a_to+9) = *(a_ij_pntr+9);
    *(a_to+10) = *(a_ij_pntr+10);
    *(a_to+11) = *(a_ij_pntr+11);

    a_to += 12;
  }
}

void PackMatrixB( int k, double *b, int ldb, double *b_to )
{
  int i;
  double 
    *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
    *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  for( i=0; i<k; i++){  /* loop over rows of B */
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

typedef union 
{
  __m256d v;
  double d[4];
} v4df_t;  

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */

  int p;
  v4df_t
    c_00_c_30_vreg, c_01_c_31_vreg,  c_02_c_32_vreg, c_03_c_33_vreg,
    a_0p_a_3p_vreg,
    b_pi_vreg;

  c_00_c_30_vreg.v = _mm256_loadu_pd((double *) &(C(0, 0))); 
  c_01_c_31_vreg.v = _mm256_loadu_pd((double *) &(C(0, 1)));
  c_02_c_32_vreg.v = _mm256_loadu_pd((double *) &(C(0, 2))); 
  c_03_c_33_vreg.v = _mm256_loadu_pd((double *) &(C(0, 3))); 

  for ( p=0; p<k; p++ ){
    a_0p_a_3p_vreg.v = _mm256_load_pd( (double *) a );
    
    b_pi_vreg.v = _mm256_broadcast_sd( (double *) b );       /* load and broadcast */
    c_00_c_30_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_00_c_30_vreg.v);

    b_pi_vreg.v = _mm256_broadcast_sd( (double *) (b+1) );   /* load and broadcast */
    c_01_c_31_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_01_c_31_vreg.v);

    b_pi_vreg.v = _mm256_broadcast_sd( (double *) (b+2) );   /* load and broadcast */
    c_02_c_32_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_02_c_32_vreg.v);

    b_pi_vreg.v = _mm256_broadcast_sd( (double *) (b+3) );   /* load and broadcast */
    c_03_c_33_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_03_c_33_vreg.v);

    a += 4;
    b += 4;
  }

  _mm256_storeu_pd(&(C(0, 0)), c_00_c_30_vreg.v);
  _mm256_storeu_pd(&(C(0, 1)), c_01_c_31_vreg.v);
  _mm256_storeu_pd(&(C(0, 2)), c_02_c_32_vreg.v);
  _mm256_storeu_pd(&(C(0, 3)), c_03_c_33_vreg.v);
}

void AddDot12x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  int p;
  v4df_t
    c_00_c_30_vreg, c_01_c_31_vreg,  c_02_c_32_vreg, c_03_c_33_vreg,
    c_40_c_70_vreg, c_41_c_71_vreg,  c_42_c_72_vreg, c_43_c_73_vreg,
    c_80_c_110_vreg, c_81_c_111_vreg,  c_82_c_112_vreg, c_83_c_113_vreg,
    a_0p_a_3p_vreg, a_4p_a_7p_vreg, a_8p_a_11p_vreg,
    b_pi_vreg;

  c_00_c_30_vreg.v = _mm256_loadu_pd((double *) &(C(0, 0))); 
  c_01_c_31_vreg.v = _mm256_loadu_pd((double *) &(C(0, 1)));
  c_02_c_32_vreg.v = _mm256_loadu_pd((double *) &(C(0, 2))); 
  c_03_c_33_vreg.v = _mm256_loadu_pd((double *) &(C(0, 3))); 
  c_40_c_70_vreg.v = _mm256_loadu_pd((double *) &(C(4, 0))); 
  c_41_c_71_vreg.v = _mm256_loadu_pd((double *) &(C(4, 1)));
  c_42_c_72_vreg.v = _mm256_loadu_pd((double *) &(C(4, 2))); 
  c_43_c_73_vreg.v = _mm256_loadu_pd((double *) &(C(4, 3))); 
  c_80_c_110_vreg.v = _mm256_loadu_pd((double *) &(C(8, 0))); 
  c_81_c_111_vreg.v = _mm256_loadu_pd((double *) &(C(8, 1)));
  c_82_c_112_vreg.v = _mm256_loadu_pd((double *) &(C(8, 2))); 
  c_83_c_113_vreg.v = _mm256_loadu_pd((double *) &(C(8, 3))); 

  for ( p=0; p<k; p++ ){
    a_0p_a_3p_vreg.v = _mm256_loadu_pd( (double *) a );
    a_4p_a_7p_vreg.v = _mm256_loadu_pd( (double *) (a+4) );
    a_8p_a_11p_vreg.v = _mm256_loadu_pd( (double *) (a+8) );
    
    b_pi_vreg.v = _mm256_broadcast_sd( (double *) b );       /* load and broadcast */
    c_00_c_30_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_00_c_30_vreg.v);
    c_40_c_70_vreg.v = _mm256_fmadd_pd(a_4p_a_7p_vreg.v, b_pi_vreg.v, c_40_c_70_vreg.v);
    c_80_c_110_vreg.v = _mm256_fmadd_pd(a_8p_a_11p_vreg.v, b_pi_vreg.v, c_80_c_110_vreg.v);


    b_pi_vreg.v = _mm256_broadcast_sd( (double *) (b+1) );   /* load and broadcast */
    c_01_c_31_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_01_c_31_vreg.v);
    c_41_c_71_vreg.v = _mm256_fmadd_pd(a_4p_a_7p_vreg.v, b_pi_vreg.v, c_41_c_71_vreg.v);
    c_81_c_111_vreg.v = _mm256_fmadd_pd(a_8p_a_11p_vreg.v, b_pi_vreg.v, c_81_c_111_vreg.v);

    b_pi_vreg.v = _mm256_broadcast_sd( (double *) (b+2) );   /* load and broadcast */
    c_02_c_32_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_02_c_32_vreg.v);
    c_42_c_72_vreg.v = _mm256_fmadd_pd(a_4p_a_7p_vreg.v, b_pi_vreg.v, c_42_c_72_vreg.v);
    c_82_c_112_vreg.v = _mm256_fmadd_pd(a_8p_a_11p_vreg.v, b_pi_vreg.v, c_82_c_112_vreg.v);

    b_pi_vreg.v = _mm256_broadcast_sd( (double *) (b+3) );   /* load and broadcast */
    c_03_c_33_vreg.v = _mm256_fmadd_pd(a_0p_a_3p_vreg.v, b_pi_vreg.v, c_03_c_33_vreg.v);
    c_43_c_73_vreg.v = _mm256_fmadd_pd(a_4p_a_7p_vreg.v, b_pi_vreg.v, c_43_c_73_vreg.v);
    c_83_c_113_vreg.v = _mm256_fmadd_pd(a_8p_a_11p_vreg.v, b_pi_vreg.v, c_83_c_113_vreg.v);

    a += 12;
    b += 4;
  }

  _mm256_storeu_pd(&(C(0, 0)), c_00_c_30_vreg.v);
  _mm256_storeu_pd(&(C(0, 1)), c_01_c_31_vreg.v);
  _mm256_storeu_pd(&(C(0, 2)), c_02_c_32_vreg.v);
  _mm256_storeu_pd(&(C(0, 3)), c_03_c_33_vreg.v);
  _mm256_storeu_pd(&(C(4, 0)), c_40_c_70_vreg.v);
  _mm256_storeu_pd(&(C(4, 1)), c_41_c_71_vreg.v);
  _mm256_storeu_pd(&(C(4, 2)), c_42_c_72_vreg.v);
  _mm256_storeu_pd(&(C(4, 3)), c_43_c_73_vreg.v);
  _mm256_storeu_pd(&(C(8, 0)), c_80_c_110_vreg.v);
  _mm256_storeu_pd(&(C(8, 1)), c_81_c_111_vreg.v);
  _mm256_storeu_pd(&(C(8, 2)), c_82_c_112_vreg.v);
  _mm256_storeu_pd(&(C(8, 3)), c_83_c_113_vreg.v);
}