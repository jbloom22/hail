package is.hail.rest

import is.hail.variant.VariantDataset
import org.http4s.server.blaze.BlazeBuilder

object ServerLinreg {
  def apply(vds: VariantDataset, covariates: Array[String], port: Int, maxWidth: Int, hardLimit: Int) {

    val restService = new ServiceLinreg(vds, covariates, maxWidth, hardLimit)
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}