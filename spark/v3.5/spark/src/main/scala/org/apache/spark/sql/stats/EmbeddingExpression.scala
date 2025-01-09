/*
 *
 *  * Licensed to the Apache Software Foundation (ASF) under one
 *  * or more contributor license agreements.  See the NOTICE file
 *  * distributed with this work for additional information
 *  * regarding copyright ownership.  The ASF licenses this file
 *  * to you under the Apache License, Version 2.0 (the
 *  * "License"); you may not use this file except in compliance
 *  * with the License.  You may obtain a copy of the License at
 *  *
 *  *   http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing,
 *  * software distributed under the License is distributed on an
 *  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  * KIND, either express or implied.  See the License for the
 *  * specific language governing permissions and limitations
 *  * under the License.
 *
 */

package org.apache.spark.sql.stats

import org.apache.curator.shaded.com.google.common.primitives.Floats
import org.apache.iceberg.spark.SparkSchemaUtil
import org.apache.iceberg.types.Types
import org.apache.spark.sql.catalyst.expressions.codegen.{CodegenContext, CodegenFallback, ExprCode}
import org.apache.spark.sql.catalyst.expressions.{Expression, UnaryExpression}
import org.apache.spark.sql.catalyst.trees.UnaryLike
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, FloatType}
import org.apache.spark.unsafe.types.UTF8String

import java.nio.ByteBuffer

/**
 * EmbeddingExpression generates embedding.
 * @param child
 */
case class EmbeddingExpression(child: Expression) extends UnaryExpression with CodegenFallback {

  override def dataType: DataType = ArrayType(FloatType, containsNull = false)

  // This expression does not allow null inputs
  override def nullable: Boolean = false

  def this(colName: String) = {
    this(col(colName).expr)
  }

  // Evaluate the embedding for a non-null input value
  override def nullSafeEval(input: Any): Any = {
    val value: String = input match {
      case utf8: UTF8String => utf8.toString
      case i: Integer       => i.toString
      case s: String        => s
      case other            => other.toString // Fallback for other types
    }
    if (value != null) {
      val embedding = Array(
        value.length.toFloat,
        value.hashCode.toFloat,
        (value.length * 2).toFloat
      )
      // Convert the float array to ArrayData
      ArrayData.toArrayData(embedding)
    }
  }

  override protected def withNewChildInternal(newChild: Expression): Expression = {
    copy(child = newChild)
  }
}
