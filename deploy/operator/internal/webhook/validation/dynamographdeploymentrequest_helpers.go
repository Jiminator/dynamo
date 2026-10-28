/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// dynamoGraphDeploymentRequestV1beta2Errors decodes the conversion carrier before typed validation.
// raw and fldPath must not be nil; oldRaw is nil when there is no previous v1beta2 spec.
// Carrier decode failures use fldPath; decoded fields use the submitted v1beta2 API's spec path.
func dynamoGraphDeploymentRequestV1beta2Errors(
	raw *apiextensionsv1.JSON,
	oldRaw *apiextensionsv1.JSON,
	fldPath *field.Path,
) field.ErrorList {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(raw.Raw, &spec); err != nil {
		return field.ErrorList{field.Invalid(fldPath, "<redacted>", "must be a valid v1beta2 spec: "+err.Error())}
	}
	if oldRaw == nil {
		return validateDynamoGraphDeploymentRequestSpecV1beta2(&spec, field.NewPath("spec"))
	}

	// Only typed old and new specs participate in the whole-spec schema ratchet.
	var oldSpec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(oldRaw.Raw, &oldSpec); err != nil {
		return field.ErrorList{field.InternalError(fldPath, fmt.Errorf("decode previous v1beta2 spec: %w", err))}
	}
	return validateDynamoGraphDeploymentRequestSpecUpdateV1beta2(&spec, &oldSpec, field.NewPath("spec"))
}

// recommendationInputErrors checks the lowered input after structural API validation.
// spec and fldPath must not be nil. A lowering failure prevents only the dependent schema check.
func recommendationInputErrors(spec *nvidiacomv1beta2.NativeSearchInput, fldPath *field.Path) field.ErrorList {
	input, err := aisimulate.Lower(spec)
	if err != nil {
		return field.ErrorList{field.Invalid(fldPath, "<redacted>", "cannot be lowered to the public AISimulate recommendation input: "+err.Error())}
	}
	if err := aisimulate.ValidateInput(input); err != nil {
		var schemaErr *jsonschema.ValidationError
		if !errors.As(err, &schemaErr) {
			return field.ErrorList{field.InternalError(fldPath, err)}
		}
		return inputSchemaErrors(schemaErr, fldPath)
	}
	return nil
}

// inputSchemaErrors reports native JSON pointers at the shared spec owner. Lowering
// combines typed fields and overrides, so a native path need not name an override.
// Do not include input values in errors: arbitrary override strings may contain secrets.
func inputSchemaErrors(err *jsonschema.ValidationError, fldPath *field.Path) field.ErrorList {
	switch err.ErrorKind.(type) {
	case *kind.Schema, *kind.Group, *kind.Reference:
		allErrs := make(field.ErrorList, 0, len(err.Causes))
		for _, cause := range err.Causes {
			allErrs = append(allErrs, inputSchemaErrors(cause, fldPath)...)
		}
		slices.SortFunc(allErrs, func(a, b *field.Error) int { return cmp.Compare(a.Detail, b.Detail) })
		return allErrs
	}

	// Keep unions as one error, but retain the paths needed to repair nested input.
	path := inputJSONPointer(err.InstanceLocation)
	keyword := strings.Join(err.ErrorKind.KeywordPath(), "/")
	detail := fmt.Sprintf("generated AISimulate input at %s violates JSON Schema keyword %s", path, keyword)
	switch constraint := err.ErrorKind.(type) {
	case *kind.Required:
		names := slices.Sorted(slices.Values(constraint.Missing))
		detail += ": missing " + strings.Join(names, ", ")
	case *kind.AdditionalProperties:
		names := slices.Sorted(slices.Values(constraint.Properties))
		detail += ": unknown " + strings.Join(names, ", ")
	case *kind.AnyOf, *kind.OneOf:
		locations := map[string]bool{}
		pending := append([]*jsonschema.ValidationError(nil), err.Causes...)
		for len(pending) > 0 {
			cause := pending[0]
			pending = append(pending[1:], cause.Causes...)
			if len(cause.InstanceLocation) > len(err.InstanceLocation) {
				locations[inputJSONPointer(cause.InstanceLocation)] = true
			}
		}
		paths := make([]string, 0, len(locations))
		for location := range locations {
			paths = append(paths, location)
		}
		slices.Sort(paths)
		if len(paths) > 0 {
			detail += "; alternatives reject " + strings.Join(paths, ", ")
		}
	}
	return field.ErrorList{field.Invalid(fldPath, "<redacted>", detail)}
}

func inputJSONPointer(location []string) string {
	escape := strings.NewReplacer("~", "~0", "/", "~1")
	tokens := make([]string, len(location))
	for i, token := range location {
		tokens[i] = escape.Replace(token)
	}
	return "/" + strings.Join(tokens, "/")
}

// invalidDynamoGraphDeploymentRequestError converts allErrs for request into an API error.
// request must not be nil.
func invalidDynamoGraphDeploymentRequestError(
	request *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	allErrs field.ErrorList,
) error {
	if len(allErrs) == 0 {
		return nil
	}
	return k8serrors.NewInvalid(nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK.GroupKind(), request.Name, allErrs)
}

func hasManualDGDRHardware(hardware *nvidiacomv1beta1.HardwareSpec) bool {
	return hardware != nil &&
		(hardware.GPUSKU != "" || hardware.VRAMMB != nil || hardware.NumGPUsPerNode != nil)
}

func isImmutableDGDRPhase(phase nvidiacomv1beta1.DGDRPhase) bool {
	switch phase {
	case nvidiacomv1beta1.DGDRPhaseProfiling,
		nvidiacomv1beta1.DGDRPhaseDeploying,
		nvidiacomv1beta1.DGDRPhaseReady,
		nvidiacomv1beta1.DGDRPhaseDeployed:
		return true
	default:
		return false
	}
}
