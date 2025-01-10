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

import org.apache.iceberg.spark.actions.{Embedding, EmbeddingBuffer}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.trees.UnaryLike
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{BinaryType, DataType}
import org.apache.spark.unsafe.types.UTF8String

case class EmbeddingAggregate(
                               child: Expression,
                               mutableAggregationBufferOffset: Int = 0,
                               inputAggregationBufferOffset: Int = 0) extends TypedImperativeAggregate[EmbeddingBuffer] with UnaryLike[Expression] {


  override protected val mutableAggBufferOffset: Int = 0
  override protected val inputAggBufferOffset: Int = 0

  def this(colName: String) = {
    this(col(colName).expr, 0, 0)
  }

  override def dataType: DataType = BinaryType

  override def nullable: Boolean = false

  override def createAggregationBuffer(): EmbeddingBuffer = {
    new EmbeddingBuffer()
  }

  // Update the aggregation buffer with the embedding for the current row
  override def update(buffer: EmbeddingBuffer, input: InternalRow): EmbeddingBuffer = {
    val value: String = child.eval(input) match {
      case utf8: UTF8String => utf8.toString
      case i: Integer => i.toString
      case s: String => s
      case _ => null
    }
    if (value != null) {
      val embedding: Array[Float] = Array(
        value.length.toFloat,
        value.hashCode.toFloat,
        (value.length * 2).toFloat
      )
      buffer.addEmbedding(new Embedding(value, embedding))
    }
    buffer
  }

  // Merge two aggregation buffers during distributed execution
  override def merge(buffer: EmbeddingBuffer, other: EmbeddingBuffer): EmbeddingBuffer = {
    other.getEmbeddingList.forEach(embedding => buffer.addEmbedding(embedding))
    buffer
  }

  override def eval(buffer: EmbeddingBuffer): Any = {
    buffer.serialize()
  }

  override def serialize(buffer: EmbeddingBuffer): Array[Byte] = {
    buffer.serialize()
  }

  override def deserialize(bytes: Array[Byte]): EmbeddingBuffer = {
    EmbeddingBuffer.deserialize(bytes)
  }

  override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate = {
    copy(mutableAggregationBufferOffset = newMutableAggBufferOffset)
  }

  override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate = {
    copy(inputAggregationBufferOffset = newInputAggBufferOffset)
  }

  override protected def withNewChildInternal(newChild: Expression): Expression = {
    copy(child = newChild)
  }
}
