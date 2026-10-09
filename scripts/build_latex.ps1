param([ValidateSet('all','report','slides')][string]$Target = 'all')
$ErrorActionPreference = 'Stop'
$env:LC_ALL = 'C'
$env:LC_CTYPE = 'C'
$env:LANG = 'C'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pdfDirectory = Join-Path $projectRoot 'output\pdf'
New-Item -ItemType Directory -Force -Path $pdfDirectory | Out-Null

function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

if ($Target -in @('all','report')) {
    $reportBuild = Join-Path $projectRoot 'outputs\latex-build\report'
    New-Item -ItemType Directory -Force -Path (Join-Path $reportBuild 'sections') | Out-Null
    Push-Location (Join-Path $projectRoot 'manuscript')
    try {
        $reportArgs = @('-halt-on-error','-interaction=nonstopmode','-no-shell-escape',"-output-directory=$reportBuild",'main.tex')
        Run-Checked 'xelatex' $reportArgs
        Run-Checked 'biber' @((Join-Path $reportBuild 'main'))
        Run-Checked 'xelatex' $reportArgs
        Run-Checked 'xelatex' $reportArgs
        Copy-Item -LiteralPath (Join-Path $reportBuild 'main.pdf') -Destination (Join-Path $pdfDirectory 'clipseg_report_template.pdf')
    } finally { Pop-Location }
}

if ($Target -in @('all','slides')) {
    $slideBuild = Join-Path $projectRoot 'outputs\latex-build\slides'
    New-Item -ItemType Directory -Force -Path $slideBuild | Out-Null
    Push-Location (Join-Path $projectRoot 'slides')
    try {
        $slideArgs = @('-halt-on-error','-interaction=nonstopmode','-no-shell-escape',"-output-directory=$slideBuild",'clipseg_slides.tex')
        Run-Checked 'xelatex' $slideArgs
        Run-Checked 'xelatex' $slideArgs
        Copy-Item -LiteralPath (Join-Path $slideBuild 'clipseg_slides.pdf') -Destination (Join-Path $pdfDirectory 'clipseg_slides_template.pdf')
    } finally { Pop-Location }
}
Write-Host "PDF output: $pdfDirectory"
