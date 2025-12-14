// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, FieldRef};
use arrow::datatypes::DataType::{Int32, Int64};
use datafusion_common::cast::as_int32_array;
use datafusion_common::{
    DataFusionError, Result, ScalarValue, exec_err, utils::take_function_args,
};
use datafusion_expr::Signature;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Volatility, ReturnFieldArgs};

/// <https://spark.apache.org/docs/latest/api/sql/index.html#factorial>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkFactorial {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for SparkFactorial {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkFactorial {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![Int32], Volatility::Immutable),
            aliases: vec![],
        }
    }
}

impl ScalarUDFImpl for SparkFactorial {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "factorial"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion_common::internal_err!(
            "return_type should not be called, use return_field_from_args instead"
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());
        Ok(Arc::new(Field::new(self.name(), Int64, nullable)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        spark_factorial(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

const FACTORIALS: [i64; 21] = [
    1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000,
    2432902008176640000,
];

pub fn spark_factorial(args: &[ColumnarValue]) -> Result<ColumnarValue, DataFusionError> {
    let [arg] = take_function_args("factorial", args)?;

    match arg {
        ColumnarValue::Scalar(ScalarValue::Int32(value)) => {
            let result = compute_factorial(*value);
            Ok(ColumnarValue::Scalar(ScalarValue::Int64(result)))
        }
        ColumnarValue::Scalar(other) => {
            exec_err!("`factorial` got an unexpected scalar type: {}", other)
        }
        ColumnarValue::Array(array) => match array.data_type() {
            Int32 => {
                let array = as_int32_array(array)?;

                let result: Int64Array = array.iter().map(compute_factorial).collect();

                Ok(ColumnarValue::Array(Arc::new(result)))
            }
            other => {
                exec_err!("`factorial` got an unexpected argument type: {}", other)
            }
        },
    }
}

#[inline]
fn compute_factorial(num: Option<i32>) -> Option<i64> {
    num.filter(|&v| (0..=20).contains(&v))
        .map(|v| FACTORIALS[v as usize])
}

#[cfg(test)]
mod test {
    use crate::function::math::factorial::spark_factorial;
    use arrow::array::{Int32Array, Int64Array};
    use datafusion_common::ScalarValue;
    use datafusion_common::cast::as_int64_array;
    use datafusion_expr::ColumnarValue;
    use std::sync::Arc;

    #[test]
    fn test_spark_factorial_array() {
        let input = Int32Array::from(vec![
            Some(-1),
            Some(0),
            Some(1),
            Some(2),
            Some(4),
            Some(20),
            Some(21),
            None,
        ]);

        let args = ColumnarValue::Array(Arc::new(input));
        let result = spark_factorial(&[args]).unwrap();
        let result = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected array"),
        };

        let actual = as_int64_array(&result).unwrap();
        let expected = Int64Array::from(vec![
            None,
            Some(1),
            Some(1),
            Some(2),
            Some(24),
            Some(2432902008176640000),
            None,
            None,
        ]);

        assert_eq!(actual, &expected);
    }

    #[test]
    fn test_spark_factorial_scalar() {
        let input = ScalarValue::Int32(Some(5));

        let args = ColumnarValue::Scalar(input);
        let result = spark_factorial(&[args]).unwrap();
        let result = match result {
            ColumnarValue::Scalar(ScalarValue::Int64(val)) => val,
            _ => panic!("Expected scalar"),
        };
        let actual = result.unwrap();
        let expected = 120_i64;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_factorial_nullability() {
        use arrow::array::{Int32Array, Int64Array};
        use datafusion_common::cast::as_int64_array;
        use datafusion_common::ScalarValue;
        use datafusion_expr::ColumnarValue;
        use std::sync::Arc;

        let scalar_null = ColumnarValue::Scalar(ScalarValue::Int32(None));
        let result = spark_factorial(&[scalar_null]).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Int64(val)) => {
                assert!(val.is_none(), "Expected NULL output for NULL input");
            }
            _ => panic!("Expected nullable Int64 scalar"),
        }

        let input = Int32Array::from(vec![None, None, None]);
        let args = ColumnarValue::Array(Arc::new(input));
        let result = spark_factorial(&[args]).unwrap();

        let array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected array output"),
        };

        let actual = as_int64_array(&array).unwrap();
        let expected = Int64Array::from(vec![None, None, None]);

        assert_eq!(actual, &expected);
    }


}
