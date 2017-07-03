package is.hail.rest

import is.hail.variant.VariantDataset
import org.http4s.server.blaze.BlazeBuilder

object RestServer {
  def apply(vds: VariantDataset, covariates: Array[String],
    port: Int = 8080, maxWidth: Int = 600000, hardLimit: Int = 100000) {

    val restService = new RestService(vds, covariates, maxWidth, hardLimit)
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}