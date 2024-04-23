jQuery(document).ready(function ( $ ) {

    jQuery(".switch_price_prod").click(function () {
        jQuery("[js-price-value]").html(jQuery(this).attr("data-price_label"));
        jQuery("[js-license-type]").val(jQuery(this).attr("data-type"));
        jQuery("[js-price-dropdown]").html(jQuery(this).attr("data-label"));
    });

    //Mobile preview Iframe action
    $('.btn-iframe-to-mobile-trigger').on('click', function (event) {
        event.preventDefault();
        $('.iframe-preview').addClass('iframe-preview--mobile');
    });
    $('.btn-iframe-to-desktop-trigger').on('click', function (event) {
        event.preventDefault();
        $('.iframe-preview').removeClass('iframe-preview--mobile');
    });

    //Theme submission preview iframe toggle
    $('.btn-iframe-to-preview-trigger').on('click', function (event) {
        event.preventDefault();
        $('.iframe-preview').attr('src', '//bootstrap-themes.github.io/dashboard');
    });
    $('.btn-iframe-to-details-trigger').on('click', function (event) {
        event.preventDefault();
        $('.iframe-preview').attr('src', location.origin + '/product/stripped');
    });

    //Setting initial frame
    $('#submitPreviewIframe').attr('src', location.origin + '/product/stripped');

    $('[js-handle="review-toggler"]').on('click', function (e) {
        e.preventDefault()
        $(this).tab('show')
        $(this).removeClass('active')
        $('.sub-nav-link.active').removeClass('active')
        $('.sub-nav-link[href="#reviews-tab"]').addClass('active')
        $('html, body').animate({
            scrollTop: $('.sub-nav-link[href="#reviews-tab"]').offset().top - 100
        }, 1000);
    });

    $('#billToEditable').popover({
      'placement':'left',
      'html': true,
      'content':'<div class="mb-1 d-flex justify-content-between align-items-center"><strong>Billing info is editable</strong><button style="font-size: 1.25rem;" class="close">×</button></div><span class="text-gray">Click your info and type to make edits, including adding a VAT or a company name!<span>',
      'trigger': 'manual'
    }).popover('show');

    $(document).on("click", ".popover .close" , function(){
      $(this).parents(".popover").popover('hide');
    });

    $(document).on("focus", "#billToEditable" , function(){
      $(".popover").popover('hide');
    });

});