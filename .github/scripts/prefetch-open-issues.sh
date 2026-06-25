#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
out_root="/tmp/gh-aw/agent"
issues_root="${out_root}/issues"
all_dir="${issues_root}/all"
batches_dir="${issues_root}/batches"
index_file="${out_root}/open-issues.tsv"
manifest_file="${issues_root}/batch-manifest.tsv"
raw_file="${issues_root}/open-issues.raw.json"

batch_size="${BATCH_SIZE:-25}"
issue_limit="${ISSUE_LIMIT:-1000}"

rm -rf "${issues_root}"
mkdir -p "${all_dir}" "${batches_dir}"

printf "number\ttitle\tupdated_at\tcreated_at\tlabel_names\n" > "${index_file}"
printf "batch\tissue_number\tupdated_at\tlabel_names\n" > "${manifest_file}"

# Fetch all open issues sorted by oldest update time first.
gh issue list \
  --repo "${repo}" \
  --state open \
  --limit "${issue_limit}" \
  --search "sort:updated-asc" \
  --json number,title,body,updatedAt,createdAt,url,labels,author,assignees \
  > "${raw_file}"

issue_count="$(jq 'length' "${raw_file}")"
if [[ "${issue_count}" == "0" ]]; then
  echo "No open issues found."
  exit 0
fi

count=0
while IFS= read -r issue_json; do
  count=$((count + 1))

  number="$(jq -r '.number' <<< "${issue_json}")"
  updated_at="$(jq -r '.updatedAt' <<< "${issue_json}")"
  created_at="$(jq -r '.createdAt' <<< "${issue_json}")"
  title="$(jq -r '.title' <<< "${issue_json}" | tr '\t\n' ' ' | sed 's/  */ /g')"
  labels="$(jq -r '[.labels[].name] | join(",")' <<< "${issue_json}")"

  printf '%s\n' "${issue_json}" > "${all_dir}/${number}.json"
  printf "%s\t%s\t%s\t%s\t%s\n" "${number}" "${title}" "${updated_at}" "${created_at}" "${labels}" >> "${index_file}"

  batch_index=$(((count - 1) / batch_size + 1))
  batch_name="$(printf 'batch-%03d' "${batch_index}")"
  batch_path="${batches_dir}/${batch_name}"
  mkdir -p "${batch_path}"
  cp "${all_dir}/${number}.json" "${batch_path}/${number}.json"

  printf "%s\t%s\t%s\t%s\n" "${batch_name}" "${number}" "${updated_at}" "${labels}" >> "${manifest_file}"
done < <(jq -c '.[]' "${raw_file}")

batch_count="$(find "${batches_dir}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"

echo "Prescanned ${count} open issues into ${all_dir}"
echo "Created ${batch_count} batch folder(s) in ${batches_dir} (batch size: ${batch_size})"
echo "Index file: ${index_file}"
echo "Batch manifest: ${manifest_file}"
