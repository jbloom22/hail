package is.hail.rest

import breeze.linalg.DenseMatrix
import is.hail.variant.VariantDataset
import org.http4s.server.blaze.BlazeBuilder

object PhenotypeTable {
  def apply(vds: VariantDataset, covariates: Array[String]): PhenotypeTable = ???
  
  // check these exist
}

case class PhenotypeTable(samples: Array[String], phenotypes: Array[String], data: DenseMatrix[Double]) {
  def selectCovariates(covariates: Array[String]): PhenotypeTable = ???
  
  def selectSamples(samples: Array[String]): PhenotypeTable = ???
}

object RestServer {
  def apply(vds: VariantDataset, covariates: Array[String], port: Int = 8080, maxWidth: Int = 600000, hardLimit: Int = 100000) {
    val phenoTable = PhenotypeTable(vds, covariates)
    val restService = new RestService(vds, phenoTable, maxWidth, hardLimit)
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}